from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
class MerkezTakipleme:
    def __init__(self, m_kaybolus=10, m_uzaklik=40):

        self.siradakiNesneID = 0
        self.nesneler = OrderedDict()
        self.kayboluslar = OrderedDict()
        self.m_kaybolus = m_kaybolus
        self.m_uzaklik = m_uzaklik

    def kaydedici(self, merkez):
        self.nesneler[self.siradakiNesneID] = merkez
        self.kayboluslar[self.siradakiNesneID] = 0
        self.siradakiNesneID += 1

    def kayit_silici(self, nesneID):
        del self.nesneler[nesneID]
        del self.kayboluslar[nesneID]

    def guncelleyici(self, rects):
        if len(rects) == 0:
            for nesneID in list(self.kayboluslar.keys()):
                self.kayboluslar[nesneID] += 1
                if self.kayboluslar[nesneID] > self.m_kaybolus:
                    self.kayit_silici(nesneID)
            return self.nesneler

        merkezGirdiler = np.zeros((len(rects), 2), dtype="int")
        for (i, (s_x, s_y, b_x, b_y)) in enumerate(rects):
            m_x = int((s_x + b_x) / 2.0)
            m_y = int((s_y + b_y) / 2.0)
            merkezGirdiler[i] = (m_x, m_y)

        if len(self.nesneler) == 0:
            for i in range(0, len(merkezGirdiler)):
                self.kaydedici(merkezGirdiler[i])
        else:
            nesneIDler = list(self.nesneler.keys())
            nesneMerkezler = list(self.nesneler.values())

            D_dist = dist.cdist(np.array(nesneMerkezler), merkezGirdiler)
            satirlar = D_dist.min(axis=1).argsort()
            sutunlar = D_dist.argmin(axis=1)[satirlar]
            u_satirlar = set()
            u_sutunlar = set()

            for(satir, sutun) in zip(satirlar, sutunlar):
                if satir in u_satirlar or sutun in u_sutunlar:
                    continue
                if D_dist[satir, sutun] > self.m_uzaklik:
                    continue
                nesneID = nesneIDler[satir]
                self.nesneler[nesneID] = merkezGirdiler[sutun]
                self.kayboluslar[nesneID] = 0
                u_satirlar.add(satir)
                u_sutunlar.add(sutun)

            un_satirlar = set(range(0, D_dist.shape[0])).difference(u_satirlar)
            un_sutunlar = set(range(0, D_dist.shape[1])).difference(u_sutunlar)

            if D_dist.shape[0] >= D_dist.shape[1]:
                for u_s in un_satirlar:
                    nesneID = nesneIDler[u_s]
                    self.kayboluslar[nesneID] += 1
                    if self.kayboluslar[nesneID] > self.m_kaybolus:
                        self.kayit_silici(nesneID)
            else:
                for sutun in un_sutunlar:
                    self.kaydedici(merkezGirdiler[sutun])

        return self.nesneler
