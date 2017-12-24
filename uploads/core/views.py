from django.shortcuts import render, redirect
from django.conf import settings
from django.core.files.storage import FileSystemStorage

from uploads.core.models import Document
from uploads.core.forms import DocumentForm
import classification
from models import Plants
from models import AllPlants
from models import PlantSearchOccur
import sqlite3

fileDes = open('/home/msd/PlantIdentificationSystem/uploads/core/static/allClassNumAndName.txt')

allClasses = {}

for i in range(0,1000):
	token = fileDes.readline()
	classId = token.split()[0]
	name1 = token.split()[1]
	name2 = token.split()[2]	
	allClasses[classId] = str(name1) + str(' ') + str(name2)
 
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


	
        absoulutePath = '/home/msd/PlantIdentificationSystem/'
        a = classification.dataFusionController(absoulutePath+uploaded_file_url,absoulutePath+uploaded_file_url2,absoulutePath+uploaded_file_url3)
		
        firstResultClassId = a[0][0]
        firstResultPlantName = str(allClasses.get(str(a[0][0])))
        resultStr = []
        count = 0
        for i in a:
                count += 1
                resultStr.append(str(count) + str(')  ') + str(allClasses.get(str(i[0]))))
             
                print(i[0],i[1])
                         
        

        #print(resultStr)
        conna = sqlite3.connect('db.sqlite3')
        database = conna.cursor()
        #print(firstResultClassId)
        #print(firstResultPlantName)
        #read = database.execute('SELECT * FROM core_allplants WHERE classId=?',(int(firstResultClassId),)) 
        read2 = database.execute('SELECT * FROM core_plantsearchoccur WHERE plantName=?',(firstResultPlantName,))
        #print('read2',read2.fetchall()[0][1],len(read2.fetchall()))     
        occuranceResults = read2.fetchall()
        #print(firstResultClassId,read.fetchall(),k)
        database.execute('''UPDATE core_plantsearchoccur SET occurance = ? WHERE plantName = ?''', (int(occuranceResults[0][2])+1, occuranceResults[0][1]))

        conna.commit()
        conna.close()
        return render(request, 'core/simple_upload.html', {
            'uploaded_file_url': resultStr, 'plants':Plants.objects.all(), 'plantsSearchResult':PlantSearchOccur.objects.order_by('-occurance')[:30] 
        })

    return render(request, 'core/simple_upload.html', {
             'plants':Plants.objects.all(), 'plantsSearchResult':PlantSearchOccur.objects.order_by('-occurance')[:30] 
        })


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


def plants_post_list(request):
	plants = Plants.objects.all()

	return render(request, 'core/simple_upload.html', {"plants":plants})



