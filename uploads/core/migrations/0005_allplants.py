# -*- coding: utf-8 -*-
# Generated by Django 1.11 on 2017-12-24 22:17
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0004_plants'),
    ]

    operations = [
        migrations.CreateModel(
            name='AllPlants',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('classId', models.IntegerField()),
                ('plantName', models.CharField(max_length=50)),
            ],
        ),
    ]
