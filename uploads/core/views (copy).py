from django.shortcuts import render, redirect
from django.conf import settings
from django.core.files.storage import FileSystemStorage

from uploads.core.models import Document
from uploads.core.forms import DocumentForm
import classification

def home(request):
    documents = Document.objects.all()
    return render(request, 'core/home.html', { 'documents': documents })


def simple_upload(request):
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
	myfile2 = request.FILES['myfile2']
	myfile3 = request.FILES['myfile3']
        fs = FileSystemStorage()
        filename = fs.save('flower.jpg', myfile)
	filename2 = fs.save('leaf.jpg',myfile2)
	filename3 = fs.save('entire.jpg',myfile3)

        uploaded_file_url = fs.url(filename)
	uploaded_file_url2 = fs.url(filename2)
	uploaded_file_url3 = fs.url(filename3)


	
	absoulutePath = '/home/msd/simple-file-upload/'
	a = classification.dataFusionController(absoulutePath+uploaded_file_url,absoulutePath+uploaded_file_url2,absoulutePath+uploaded_file_url3)

        return render(request, 'core/simple_upload.html', {
            'uploaded_file_url': a
        })

    return render(request, 'core/simple_upload.html')


def model_form_upload(request):
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('home')
    else:
        form = DocumentForm()
    return render(request, 'core/model_form_upload.html', {
        'form': form
    })
