import cv2
import numpy as np
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
from Merkez_Takipleme import MerkezTakipleme
from Kisi_Nesne import KisiTakipleme

model = cv2.dnn.readNetFromDarknet("yolov3.cfg", "kafaModeli.weights")
model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
model.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
print("modeller bulundu")
nesneler = ["kafa"]
yayin = cv2.VideoCapture(0)
print("yayin basladi")
W = None
H = None
mt = MerkezTakipleme(m_kaybolus=6, m_uzaklik=90)
izlenen_kisiler = {}
toplam_cikan = 0
toplam_giren = 0
uygun_guven_degeri=0.6
k_ust_sinir = 120
k_alt_sinir = 380

def insan_sayac(kisiler):
    an_H = an.shape[0]
    an_W = an.shape[1]
    ust_sinir = 120
    alt_sinir = 380
    global toplam_cikan
    global toplam_giren
    global toplam_giren_cikan
    for (kisi_id, merkez) in kisiler.items():
        kisi = izlenen_kisiler.get(kisi_id, None)
        if kisi is None:
            kisi = KisiTakipleme(kisi_id, merkez)

        else:
            y = [c[1] for c in kisi.kisiNesneMerkezi]
            yon = merkez[1] - np.mean(y)
            kisi.kisiNesneMerkezi.append(merkez)
            if not kisi.kisiNesneSayildiKontrol:
                if yon < 0 and merkez[1] in range(an_H // 2 - 10, an_H // 2 + 10):
                    toplam_giren += 1
                    kisi.kisiNesneSayildiKontrol = True
                elif yon > 0 and merkez[1] in range(an_H // 2 - 10, an_H // 2 + 10):
                    toplam_cikan += 1
                    kisi.kisiNesneSayildiKontrol = True
            #if kisi.kisiNesneSayildiKontrol==True:
                #if merkez[1] not in range(an_H // 2 - 10, an_H // 2 + 10):
                    #kisi.kisiNesneSayildiKontrol= False
            #if merkez[1]<ust_sinir and kisi.kisiNesneSayildiKontrol==True and yon<0:
                #print("ust sınırda")
                #mt.kayit_silici(kisi_id)

            #if merkez[1]>alt_sinir and kisi.kisiNesneSayildiKontrol==True and yon>0:
                #print("alt sınırda")
                #mt.kayit_silici(kisi_id)

        izlenen_kisiler[kisi_id] = kisi
        cv2.circle(an, (merkez[0], merkez[1]), 4, (0, 255, 0), -1)
        id_bilgi = "kisi {}".format(kisi_id)
        cv2.putText(an, id_bilgi, (merkez[0] - 10, merkez[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)


while 1:
    h_an, an = yayin.read()
    an = cv2.resize(an, (500, 500))
    an_h = an.shape[0]
    an_w = an.shape[1]
    rects = []
    blob = cv2.dnn.blobFromImage(an, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    model.setInput(blob)
    output_layers_name = model.getUnconnectedOutLayersNames()
    layerOutputs = model.forward(output_layers_name)

    boxes = []
    guven_degeri_list = []
    nesne_id_list = []
    for output in layerOutputs:
        for tespit in output:
            score = tespit[5:]
            class_id = np.argmax(score)
            guven_degeri = score[class_id]
            if guven_degeri > uygun_guven_degeri:
                merkez_x = int(tespit[0] * an_w)
                merkez_y = int(tespit[1] * an_h)
                w = int(tespit[2] * an_w)
                h = int(tespit[3] * an_h)
                x = int(merkez_x - w / 2)
                y = int(merkez_y - h / 2)
                boxes.append([x, y, w, h])
                guven_degeri_list.append((float(guven_degeri)))
                nesne_id_list.append(class_id)

                #if merkez_y>k_ust_sinir and merkez_y<k_alt_sinir:
                    #w = int(tespit[2] * an_w)
                    #h = int(tespit[3] * an_h)
                    #x = int(merkez_x - w / 2)
                    #y = int(merkez_y - h / 2)
                    #boxes.append([x, y, w, h])
                    #guven_degeri_list.append((float(guven_degeri)))
                    #nesne_id_list.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, guven_degeri_list, .5, .4)

    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            if nesne_id_list[i] == 0:
                rects.append((x, y, x + w, y + h))
                kisiler = mt.guncelleyici(rects)
                insan_sayac(kisiler)

    font = cv2.FONT_HERSHEY_PLAIN
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            nesne_etiket = str(nesneler[nesne_id_list[i]])
            guven_degeri = str(round(guven_degeri_list[i], 2))

            cv2.rectangle(an, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(an, nesne_etiket + " " + guven_degeri, (x, y + 50), font, 2, (255, 0, 255), 2)
    bilgi = [
        ("Giren Kisi Sayisi", toplam_giren),
        ("Cikan Kisi Sayisi", toplam_cikan),
        ("Icerideki Kisi Sayisi", toplam_giren - toplam_cikan)

    ]

    for (i, (k, v)) in enumerate(bilgi):
        text = "{}: {}".format(k, v)
        cv2.putText(an, text, (10, an_h - ((i * 20) + 20)),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    cv2.putText(an, "Kapi Esigi Cizgi", (10, (an_h // 2) + 30),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.line(an, (0, an_h // 2), (an_w, an_h // 2), (0, 255, 255), 1)
    cv2.line(an, (0, (an_h // 2) + 10), (an_w, (an_h // 2) + 10), (255, 255, 255), 1)
    cv2.line(an, (0, (an_h // 2) - 10), (an_w, (an_h // 2) - 10), (255, 255, 255), 1)
    cv2.line(an, (0, 120), (an_w, 120), (0, 0, 255), 1)
    cv2.line(an, (0, 380), (an_w, 380), (0, 0, 255), 1)
    cv2.putText(an, "Ust Sinir Cizgi", (200, 110),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(an, "Alt Sinir Cizgi", (200, 400),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow('Bina Kisi Sayaci', an)
    if cv2.waitKey(1) == ord('q'):
        break

yayin.release()
cv2.destroyAllWindows()
