import sys  # sys нужен для передачи argv в QApplication
from PyQt5.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QLabel,
                             QApplication, QPushButton, QComboBox, QLineEdit, QTextEdit, QErrorMessage)
from PyQt5.QtGui import QPixmap
import numpy as np
from PIL import Image
import os


class ExampleApp(QWidget):
    def __init__(self):
        super().__init__()
        self.x1 = np.array([1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1])
        self.x2 = np.array([0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0])
        self.x3 = np.array([1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1])
        self.x4 = np.array([0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0])
        self.x5 = np.array([1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1])
        self.h = 3
        self.w = 4
        self.r = 0.8
        self.koef = 90
        self.xx = np.array([self.x2, self.x3, self.x4, self.x5])
        self.neur_arr = np.array([])
        self.neur_arr = np.append(Neuron(), Neuron(self.x1))
        self.create_pic()
        self.init_ui()
        self.events()

    def init_ui(self):
        self.lbl_r = QLineEdit(str(self.r))
        self.lbl_r.setToolTip('Введите число от 0 до 1')
        self.lbl1 = QLabel("Порог сходства:")
        self.obraz_box = QComboBox()
        self.exbtn = QPushButton("Выход")
        self.create_obr = QPushButton("Go")
        self.obraz_box.addItems(["Образ1", "Образ2", "Образ3", "Образ4", "Образ5"])
        self.label_pic = QLabel("")
        self.label_pic.setPixmap(QPixmap(r'pictures/img1.bmp'))
        self.logger = QTextEdit()
        self.logger.setReadOnly(True)
        layout1 = QHBoxLayout()
        layout2 = QVBoxLayout()
        layout3 = QHBoxLayout()

        layout1.addWidget(self.lbl1)
        layout1.addWidget(self.lbl_r)
        layout1.addWidget(self.create_obr)

        layout2.addLayout(layout1)

        layout2.addWidget(self.label_pic)

        layout2.addWidget(self.obraz_box)
        layout2.addWidget(self.exbtn)

        layout3.addLayout(layout2)
        layout3.addWidget(self.logger)
        self.setLayout(layout3)

        self.setGeometry(100, 50, 1200, 600)

        self.show()

    def quit(self):
        for i in range(len(self.neur_arr)):
            path = os.path.join(os.path.abspath(os.path.dirname(__file__)), r'pictures\imgnew{0}.bmp'.format(i+1))
            os.remove(path)
        exit()

    def create_pic(self):
        img1 = Image.new("RGB", (self.w * self.koef, self.h * self.koef))
        img2 = Image.new("RGB", (self.w * self.koef, self.h * self.koef))
        img3 = Image.new("RGB", (self.w * self.koef, self.h * self.koef))
        img4 = Image.new("RGB", (self.w * self.koef, self.h * self.koef))
        img5 = Image.new("RGB", (self.w * self.koef, self.h * self.koef))

        img1 = self.load_pic(img1, self.x1)
        img2 = self.load_pic(img2, self.x2)
        img3 = self.load_pic(img3, self.x3)
        img4 = self.load_pic(img4, self.x4)
        img5 = self.load_pic(img5, self.x5)

        img1.save(r'pictures/img1.bmp', "BMP")
        img2.save(r'pictures/img2.bmp', "BMP")
        img3.save(r'pictures/img3.bmp', "BMP")
        img4.save(r'pictures/img4.bmp', "BMP")
        img5.save(r'pictures/img5.bmp', "BMP")

    def load_pic(self, img, map):
        for i in range(self.h):
            for j in range(self.w):
                if map[i * self.w + j] == 0:
                    for g in range(self.koef):
                        for t in range(self.koef):
                            img.putpixel(((j * self.koef + t), (i * self.koef + g)), (255, 255, 255))
        return img

    def events(self):
        self.exbtn.clicked.connect(self.quit)
        self.create_obr.clicked.connect(self.creative)
        self.obraz_box.activated[str].connect(self.on_activated)

    def is_num(self, string):
        try:
            float(string)
            return True
        except ValueError:
            self.show_error()
            return False

    def creative(self):
        self.logger.clear()
        for i in range(5, 10):
            self.obraz_box.removeItem(5)
        if self.is_num(self.lbl_r.text()):
            if (float(self.lbl_r.text()) < 1.0) and (float(self.lbl_r.text()) > 0.0):
                self.r = float(self.lbl_r.text())
            else:
                self.show_error()
                return
        else:
            return
        self.neur_arr = np.append(Neuron(), Neuron(self.x1))
        self.logger.append("Первый образец запоминается без изменений")
        self.logger.append("")
        num = 2
        for el in self.xx:
            self.logger.append("                                       На вход подается образец " + str(num) + "")
            self.logger.append("")
            ## --------- Распознавание ------------ ##
            self.logger.append("Распознавание:")
            self.neur_arr = np.append(self.neur_arr, Neuron(el))
            s = self.find_s()
            self.logger.append("")
            ## ---- Сравнение --- ##
            if s[0] != 0:
                self.logger.append("Сравнение:")
                c, kol = self.find_c(s[0])
                pn = kol / len(c)
                if pn > self.r:
                    self.logger.append(
                        "Результат сравнения: %.3f >" % pn + "pn=" + str(self.r) + "    => Переобучение " + str(
                            s[0]) + " нейрона")
                    self.neur_arr[s[0]].b = self.relearn(s[0], c)
                    self.neur_arr = np.delete(self.neur_arr, -1)
                else:
                    self.logger.append(
                        "Результат сравнения: %.3f <" % pn + "pn=" + str(
                            self.r) + "    => Новый нейрон для запоминания")
                self.logger.append("")
            num += 1
        self.neur_arr = np.delete(self.neur_arr, 0)
        temp = 0
        for el in self.neur_arr:
            img_new = Image.new("RGB", (self.w * self.koef, self.h * self.koef))
            img_new = self.load_pic(img_new, el.b)
            img_new.save(r'pictures/imgnew' + str(temp + 1) + ".bmp", "BMP")
            self.obraz_box.addItems(["Созданный образ" + str(temp + 1) + ""])
            temp += 1
        self.logger.append("Результат работы:")
        for i in range(0, len(self.neur_arr)):
            strin1 = "Веса B нейрона " + str(i + 1) + ":"
            self.logger.append(strin1)
            strin1 = ""
            for j in range(len(self.neur_arr[i].b)):
                strin1 += "%0.2f " % self.neur_arr[i].b[j]
            self.logger.append(strin1)

    def show_error(self):
        self.msg = QErrorMessage()
        self.msg.showMessage('Введите число от 0 до 1')

    def on_activated(self):
        if self.obraz_box.currentIndex() == 0:
            self.label_pic.setPixmap(QPixmap(r'pictures/img1.bmp'))
        if self.obraz_box.currentIndex() == 1:
            self.label_pic.setPixmap(QPixmap(r'pictures/img2.bmp'))
        if self.obraz_box.currentIndex() == 2:
            self.label_pic.setPixmap(QPixmap(r'pictures/img3.bmp'))
        if self.obraz_box.currentIndex() == 3:
            self.label_pic.setPixmap(QPixmap(r'pictures/img4.bmp'))
        if self.obraz_box.currentIndex() == 4:
            self.label_pic.setPixmap(QPixmap(r'pictures/img5.bmp'))
        if self.obraz_box.currentIndex() == 5:
            self.label_pic.setPixmap(QPixmap(r'pictures/imgnew1.bmp'))
        if self.obraz_box.currentIndex() == 6:
            self.label_pic.setPixmap(QPixmap(r'pictures/imgnew2.bmp'))
        if self.obraz_box.currentIndex() == 7:
            self.label_pic.setPixmap(QPixmap(r'pictures/imgnew3.bmp'))
        if self.obraz_box.currentIndex() == 8:
            self.label_pic.setPixmap(QPixmap(r'pictures/imgnew4.bmp'))
        if self.obraz_box.currentIndex() == 9:
            self.label_pic.setPixmap(QPixmap(r'pictures/imgnew5.bmp'))

    def find_s(self):
        maximum = 0
        s = np.array([])
        for i in range(len(self.neur_arr) - 1):
            s = np.append(s, self.neur_arr[i].signal(self.neur_arr[-1].t))

        for i in s:
            if i > maximum:
                maximum = i
        for i in range(len(s)):
            if i == 0:
                if round(s[i], 3) == round(maximum, 3):
                    self.logger.append("Выход нераспределенного нейрона Sн = %.3f(Победитель)" % s[i])
                    self.logger.append("Обучение новому образцу, поскольку победил нераспределенный нейрон")
                else:
                    self.logger.append("Выход нераспределенного нейрона Sн = %.3f" % s[i])
            else:
                if round(s[i], 3) == round(maximum, 3):
                    self.logger.append("Выход нейрона " + str(i) + " S" + str(i) + " = %.3f(Победитель)" % s[i])
                else:
                    self.logger.append("Выход нейрона " + str(i) + " S" + str(i) + " = %.3f" % s[i])
        return list(s).index(maximum), maximum

    def find_c(self, num_neur):
        c = np.array([])
        tmp = 0
        for i in range(len(self.neur_arr[num_neur].t)):
            if self.neur_arr[-1].t[i] == self.neur_arr[num_neur].t[i]:
                c = np.append(c, self.neur_arr[num_neur].t[i])
                tmp += 1
            else:
                c = np.append(c, 0)
        return c, tmp

    def relearn(self, num_neur, c):
        for i in range(len(self.neur_arr[num_neur].b)):
            self.neur_arr[num_neur].b[i] = (self.neur_arr[num_neur].lyam * c[i]) / (
                        self.neur_arr[num_neur].lyam - 1 + np.sum(c))
        return self.neur_arr[num_neur].b


class Neuron:
    def __init__(self, t=np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), r=0.8, lyam=2):
        self.t = t
        self.r = r
        self.lyam = lyam
        if np.sum(t) == len(t):
            self.b = self.b_gen(past=False)
        else:
            self.b = self.b_gen()

    def b_gen(self, past=True):
        tmp = np.array([])
        if past:
            for i in range(len(self.t)):
                tmp = np.append(tmp, (self.lyam * self.t[i] / (self.lyam - 1 + np.sum(self.t))))
        else:
            for i in range(len(self.t)):
                tmp = np.append(tmp, (self.lyam / (self.lyam - 1 + len(self.t))))
        return tmp

    def signal(self, x_out):
        s = 0
        for i in range(len(self.t)):
            s += x_out[i] * self.b[i]
        return s


def main():
    app = QApplication(sys.argv)  # Новый экземпляр QApplication
    window = ExampleApp()  # Создаём объект класса ExampleApp
    window.show()  # Показываем окно
    app.exec_()  # и запускаем приложение


if __name__ == '__main__':  # Если мы запускаем файл напрямую, а не импортируем
    main()  # то запускаем функцию main()
