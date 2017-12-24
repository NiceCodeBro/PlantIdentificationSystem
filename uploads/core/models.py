from __future__ import unicode_literals

from django.db import models


class Document(models.Model):
    description = models.CharField(max_length=255, blank=True)
    document = models.FileField(upload_to='documents/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

class Plant(models.Model):
	classId = models.IntegerField()
	plantName = models.CharField(max_length=50)
	def __str__(self):
		return str(self.classId) + " - " + self.plantName


class Plants(models.Model):
	classId = models.IntegerField()
	plantName = models.CharField(max_length=50)
	def __str__(self):
		return str(self.classId) + " - " + self.plantName
class AllPlants(models.Model):
	classId = models.IntegerField()
	plantName = models.CharField(max_length=50)
	def __str__(self):
		return str(self.classId) + " - " + self.plantName

class PlantSearchOccur(models.Model):
	plantName = models.CharField(max_length=50)
	occurance = models.IntegerField()
	def __str__(self):
		return str(self.occurance) + " - " + self.plantName
