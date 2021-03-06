from django.conf.urls import url
from django.contrib import admin
from django.conf import settings
from django.conf.urls.static import static

from uploads.core import views
from uploads.core.views import plants_post_list

urlpatterns = [
    #url(r'^$', views.home, name='home'),
    url(r'^$', views.simple_upload, name='home'),
#    url(r'^uploads/form/$', views.model_form_upload, name='model_form_upload'),
    url(r'^admin/', admin.site.urls),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
	#urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
