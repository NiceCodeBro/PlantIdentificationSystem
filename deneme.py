import sqlite3
baglanti = sqlite3.connect('db.sqlite3')


veritabani_sec = baglanti.cursor()
 

oku = veritabani_sec.execute('SELECT * FROM core_plants')
print(oku.fetchall())

print(veritabani_sec.execute('''UPDATE core_plantsearchoccur SET occurance = ? WHERE plantName = ?''', (0, "Cakile maritima")))
oku = veritabani_sec.execute('''SELECT * FROM core_plantsearchoccur WHERE plantName = ? ''',("Salvia pratensis",))
print(oku.fetchall()[0][1])

baglanti.commit()
baglanti.close()
