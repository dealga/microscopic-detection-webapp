# views.py
import os
import zipfile
import tempfile
from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponse, JsonResponse
from django.conf import settings
from django.core.files import File
from django.urls import reverse
from .models import Analysis, DetectedFigure
from .forms import TiffUploadForm
from .utils.tiff_scanner import TIFFScanner
from .utils.mitotic_counter import process_video, process_video_with_boxes, move_figure
from .utils.tiff_scanner import convert_to_mp4
from .utils.hpf_calculator import compute_mitotic_density_from_image, get_tumor_grade


from django.template.loader import render_to_string
import io
from django.utils.text import slugify

def download_hpf_report(request, analysis_id):
    analysis = get_object_or_404(Analysis, id=analysis_id)

    report_text = render_to_string('hpf_report_template.txt', {'analysis': analysis})

    filename = f"HPF_Report_{slugify(analysis.id)}.txt"
    response = HttpResponse(report_text, content_type='text/plain')
    response['Content-Disposition'] = f'attachment; filename="{filename}"'
    return response


def home(request):
    if request.method == 'POST':
        form = TiffUploadForm(request.POST, request.FILES)
        if form.is_valid():
            analysis = form.save()
            return redirect('processing', analysis_id=analysis.id)
    else:
        form = TiffUploadForm()
    
    return render(request, 'mitotic_app/home.html', {'form': form})

def processing(request, analysis_id):
    analysis = get_object_or_404(Analysis, id=analysis_id)
    
    if request.headers.get('x-requested-with') == 'XMLHttpRequest':
        # This is for AJAX status checks
        if analysis.video_file:
            progress = 50
            status = "Processing video..."
            if hasattr(analysis, 'figures'):
                if analysis.figures.count() > 0:
                    progress = 100
                    status = "Processing complete"
                    return JsonResponse({
                        'progress': progress, 
                        'status': status,
                        'redirect': reverse('results', args=[analysis_id])
                    })
        else:
            progress = 25
            status = "Creating video from TIFF image..." 
        
        return JsonResponse({'progress': progress, 'status': status})
    
    # Check if processing needs to be started
    if not analysis.video_file:
        # Convert TIFF to video
        try:
            # Create directory for this analysis
            analysis_dir = os.path.join(settings.MEDIA_ROOT, f'analysis_{analysis.id}')
            os.makedirs(analysis_dir, exist_ok=True)
            
            # Process the TIFF image
            scanner = TIFFScanner(analysis.uploaded_image.path)
            video_path = scanner.smooth_scan(output_dir=analysis_dir)
            
            # Add this section for HPF calculation
            try:
                print("Starting HPF calculation...")
                # Get scan parameters from TIFFScanner.smooth_scan
                window_size = (256, 256)
                speed = 20
                y_speed_multiplier = 12.5  # From TIFFScanner.smooth_scan
                
                hpf_data = compute_mitotic_density_from_image(
                    image_path=analysis.uploaded_image.path,
                    mitotic_count=0,  # Will be updated later after detection
                    step_x=speed,
                    step_y=int(speed * y_speed_multiplier)
                )
                
                # Store HPF data in the analysis model
                analysis.x_mpp = hpf_data['x_mpp']
                analysis.y_mpp = hpf_data['y_mpp']
                analysis.hpf_width_px, analysis.hpf_height_px = hpf_data['hpf_size']
                analysis.total_hpfs = hpf_data['total_hpfs']
                analysis.mitoses_per_10_hpf = 0  # Will be updated after figure detection
                analysis.tumor_grade = 1  # Will be updated after figure detection
                analysis.save()
                
                print(f"HPF data saved to Analysis {analysis.id}: {hpf_data}")
                
            except Exception as e:
                print(f"Error calculating HPF data: {e}")
                # Continue processing even if HPF calculation fails
            
            # Update the analysis with the video file
            rel_path = os.path.relpath(video_path, settings.MEDIA_ROOT)
            with open(video_path, 'rb') as f:
                analysis.video_file.save(os.path.basename(video_path), File(f), save=True)
                
            # Now process video to count mitotic/non-mitotic figures
            results = process_video(
                video_path=video_path,
                model_path=os.path.join(settings.BASE_DIR, 'model', 'best.pt'),
                analysis_id=analysis.id
            )
            
            if results:
                # Path to raw mp4 from YOLO output
                raw_video_path = os.path.join(settings.MEDIA_ROOT, results['processed_video'])

                # Create new filename for safe browser-compatible mp4
                base, _ = os.path.splitext(results['processed_video'])
                safe_filename = f"{base}_browser.mp4"
                safe_video_path = os.path.join(settings.MEDIA_ROOT, safe_filename)

                # Convert using ffmpeg (even if it's mp4, we re-encode)
                convert_to_mp4(raw_video_path, safe_video_path)

                # Save the converted path to the model
                analysis.processed_video.name = safe_filename
                analysis.save()
                
                # Save detected figures to database
                for figure_data in results['figures_data']:
                    DetectedFigure.objects.create(
                        analysis=analysis,
                        image_file=figure_data['image_path'],
                        category=figure_data['category'],
                        confidence=figure_data['confidence'],
                        frame_number=figure_data['frame_number']
                    )
                
                # Add this section to update HPF calculations with detected figures
                if hasattr(analysis, 'total_hpfs') and analysis.total_hpfs:
                    print("Updating HPF analysis with detected mitotic figures")
                    analysis.update_hpf_analysis()
                
                return redirect('results', analysis_id=analysis.id)
                
        except Exception as e:
            print(f"Error processing image: {e}")
            # Handle error
    
    return render(request, 'mitotic_app/processing.html', {'analysis': analysis})

def results(request, analysis_id):
    analysis = get_object_or_404(Analysis, id=analysis_id)
    
    # Get figures grouped by category
    mitotic_figures = analysis.figures.filter(category=DetectedFigure.MITOTIC).order_by('frame_number')
    non_mitotic_figures = analysis.figures.filter(category=DetectedFigure.NON_MITOTIC).order_by('frame_number')
    discarded_figures = analysis.figures.filter(category=DetectedFigure.DISCARDED).order_by('frame_number')
    
    # Add this line to ensure HPF analysis is up to date
    if hasattr(analysis, 'total_hpfs') and analysis.total_hpfs:
        analysis.update_hpf_analysis()
    
    context = {
        'analysis': analysis,
        'mitotic_figures': mitotic_figures,
        'non_mitotic_figures': non_mitotic_figures,
        'discarded_figures': discarded_figures,
        'mitotic_count': mitotic_figures.count(),
        'non_mitotic_count': non_mitotic_figures.count(),
        'total_count': mitotic_figures.count() + non_mitotic_figures.count()
    }
    
    return render(request, 'mitotic_app/results.html', context)

def move_figure_view(request, figure_id):
    if request.method == 'POST':
        new_category = request.POST.get('category')
        figure = get_object_or_404(DetectedFigure, id=figure_id)
        # Add reference to analysis
        analysis = figure.analysis
        
        if new_category in [DetectedFigure.MITOTIC, DetectedFigure.NON_MITOTIC, DetectedFigure.DISCARDED]:
            move_figure(figure_id, new_category)
            
            # Add this section to update HPF calculations after moving a figure
            if hasattr(analysis, 'total_hpfs') and analysis.total_hpfs:
                analysis.update_hpf_analysis()
                
            return JsonResponse({'status': 'success'})
    
    return JsonResponse({'status': 'error'}, status=400)

def download_figures(request, analysis_id, category=None):
    analysis = get_object_or_404(Analysis, id=analysis_id)

    with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp:
        with zipfile.ZipFile(tmp, 'w') as archive:
            figures = analysis.figures.all()
            if category:
                figures = figures.filter(category=category)
            
            for figure in figures:
                file_path = figure.image_file.path
                if os.path.exists(file_path):
                    filename = os.path.basename(file_path)
                    archive.write(file_path, filename)

            # ✅ Only add the video when downloading all
            if category is None and analysis.processed_video and os.path.exists(analysis.processed_video.path):
                video_filename = os.path.basename(analysis.processed_video.path)
                archive.write(analysis.processed_video.path, video_filename)

            # ✅ Add HPF report only when downloading all
            if category is None:
                report_text = render_to_string('hpf_report_template.txt', {'analysis': analysis})
                archive.writestr('HPF_Report.txt', report_text)

    with open(tmp.name, 'rb') as f:
        response = HttpResponse(f.read(), content_type='application/zip')
        filename = f'mitotic_analysis_{analysis_id}'
        if category:
            filename += f'_{category}'
        response['Content-Disposition'] = f'attachment; filename={filename}.zip'

    os.unlink(tmp.name)
    return response