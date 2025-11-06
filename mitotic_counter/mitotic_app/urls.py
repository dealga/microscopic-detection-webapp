# mitotic_app/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('processing/<int:analysis_id>/', views.processing, name='processing'),
    path('results/<int:analysis_id>/', views.results, name='results'),
    path('move-figure/<int:figure_id>/', views.move_figure_view, name='move_figure'),
    path('download/<int:analysis_id>/', views.download_figures, name='download_all'),
    path('download/<int:analysis_id>/<str:category>/', views.download_figures, name='download_category'),
    path('download-hpf-report/<int:analysis_id>/', views.download_hpf_report, name='download_hpf_report'),
]