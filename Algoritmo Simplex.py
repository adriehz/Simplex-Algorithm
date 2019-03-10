
import pandas as pd
import numpy as np
# Usando Python 3.0 en adelante para tkinter
import tkinter as tk
from tkinter import font, ttk

np.set_printoptions(suppress=True)
pd.set_option("display.width", 500)
class Application():

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Algoritmo Simplex")
        self.fuente = font.Font(weight="bold")

        self.num_var = tk.IntVar()
        self.num_restr = tk.IntVar()
        self.maximizar = tk.BooleanVar(value=False)
        self.minimizar = tk.BooleanVar(value=False)


        self.etiq = tk.Label(self.root, text="ALGORITMO SIMPLEX PRIMAL DUAL", font=self.fuente)
        self.etiq1 = tk.Label(self.root, text="Min/Max", font=self.fuente)
        self.etiq2 = tk.Label(self.root, text="Num. Variables",font = self.fuente)
        self.etiq3 = tk.Label(self.root, text="Num. Restricciones", font=self.fuente)

        choices = {"Minimizar","Maximizar"}
        self.varchoice = tk.StringVar()
        self.varchoice.set("Maximizar")
        self.menu = tk.OptionMenu(self.root,self.varchoice, *choices)
        self.entry1 = tk.Entry(self.root, textvariable=self.num_var,width=5)
        self.entry2 = tk.Entry(self.root, textvariable=self.num_restr,width=5)
        self.boton = tk.Button(self.root, text="Aceptar", command=self.ventana)

        self.etiq.pack(side="top",fill="both", expand=True,padx=10,pady=10)
        self.etiq1.pack(side="top", fill="both", expand=True, padx=10,pady=10)
        self.menu.pack(side="top", fill="both", expand=True, padx=10,pady=10)
        self.etiq2.pack(side="top", fill="both", expand=True, padx=10,pady=10)
        self.entry1.pack(side="top", fill="both", expand=True, padx=10,pady=10)
        self.etiq3.pack(side="top",  fill="both", expand=True, padx=10,pady=10)
        self.entry2.pack(side="top", fill="both", expand=True, padx=10,pady=10)
        self.boton.pack(side="top", fill="both", expand=True, padx=10,pady=10)

        self.root.mainloop()


    def ventana(self):
        self.totalvar =int(self.entry1.get())
        self.totalrestr = int(self.entry2.get())
        self.maxormin= self.varchoice.get()
        self.dialogo = tk.Toplevel()
        self.entries = []
        self.entries3 = []
        self.menuIneq =[]
        self.restricciones = []
        self.inequal = []
        self.labels = []
        self.variables = []
        self.resource = []

        self.marco = ttk.Frame(self.dialogo, borderwidth=2, relief="raised",padding=(10,10))


        if(self.maxormin=="Maximizar"):
            self.etiqmaxmin = tk.Label(self.dialogo, text="MAX: ",font=self.fuente)
        else:
            self.etiqmaxmin = tk.Label(self.dialogo, text="MIN: ", font=self.fuente)

        self.etiqsubject = tk.Label(self.dialogo,text="S.T.", font=self.fuente)

        for x in range(int(self.totalrestr)):
            self.inequal.append(tk.StringVar())

        col=0
        for x in range(int(self.totalvar)):
            col+=2
            self.variables.append(tk.Entry(self.dialogo))
            self.labels.append(tk.Label(self.dialogo, text="X"+str(x+1)))
            self.variables[x].grid(column= col+1,row= 0)
            self.labels[x].grid(column= col+2,row=0)

        inequality = {"<=", ">="}
        for y in range(int(self.totalrestr)):
            cols=0
            self.labels2 = []
            self.entries2 = []
            for x in range(int(self.totalvar)):
                cols+=2

                self.entries2.append(tk.Entry(self.dialogo))
                self.labels2.append(tk.Label(self.dialogo, text="X" + str(x + 1)))
                self.entries2[x].grid(column=cols + 1, row=y+2)
                self.labels2[x].grid(column=cols + 2, row=y+2)

            self.menuIneq.append(tk.OptionMenu(self.dialogo, self.inequal[y], *inequality))
            self.resource.append(tk.Entry(self.dialogo))
            self.menuIneq[y].grid(column=cols+3, row=y+2)
            self.resource[y].grid(column=cols+4, row=y+2)
            self.restricciones += self.entries2

        self.botonCalcular = tk.Button(self.dialogo, text="Calcular", command=self.calcularSimplex)
        self.botonCalcular.grid(column=int(self.totalrestr),row=int(self.totalrestr)+2)

        self.marco.grid(column=0, row=0)
        self.etiqmaxmin.grid(column=0, row=0)
        self.etiqsubject.grid(column=0, row=1)
        # Muestra el Resultado
        self.varmostrar = tk.StringVar(value="No Dejar Cuadros Vacios!")
        self.mostrar = tk.Label(self.dialogo, textvariable=self.varmostrar, state="disabled", font=self.fuente)
        self.mostrar.grid(column=0, row=self.totalrestr + 3)

        self.root.wait_window(self.dialogo)

    def calcularSimplex(self):

        lista = [float(e.get()) for e in self.restricciones]
        lista2 = [e.get() for e in self.inequal]
        lista3 = [float(e.get()) for e in self.resource]
        lista4 = [float(e.get()) for e in self.variables]
        restr= self.totalrestr
        var=self.totalvar

        mm=self.maxormin
        maximizar = False
        minimizar= False
        if (mm == "Maximizar"):
            maximizar = True
        if (mm == "Minimizar"):
            minimizar = True
        # Crear variables
        for x in range(self.totalrestr):
            lista4.append(0)
        c = np.array(lista4)
        b = np.array(lista3)
        cols =[]
        i=0
        for y in range(self.totalrestr):
            row = []
            for x in range(self.totalvar):
                row.append(lista[i])
                i += 1
            cols.append(row)
        A= np.array(cols)

        for x in range(self.totalrestr):
            if(lista2[x] == '>='):
                b[x] = (-1)*b[x]
                A[x] = (-1)*A[x]

        def Inicializar(A, b, restr, var, mm):
            P = matrixP(A, restr, var)
            B = matrixB(P, var)
            BInv = matrixBInv(B)
            b = matb(b)
            CB = matCB(restr)
            xB = matX(restr, var)
            inicio = matrixInicio(restr, var)
            Final = calcularMatrix(inicio, P, BInv, CB, b, c)

            # Checar si la matriz b tiene elementos negativos, aplicar DUAL
            ''''
            while (min(Final[1:-1, restr + var]) < 0):
                entrada, salida = buscaDual(Final, restr, var)

                xB[salida, 0] = entrada
                B[:, salida] = P[:, entrada]
                BInv = matrixBInv(B)
                CB[salida] = c[entrada]
                Final = calcularMatrix(inicio, P, BInv, CB, b, c)
                print(Final, "DUAL")'''

            if (maximizar):
                while((min(Final[0, :-1]) < 0) or (min(Final[1:-1, restr + var]) < 0)):
                    if(min(Final[1:-1, restr + var]) < 0):
                        entra, sale = buscaDual(Final, restr, var)
                    else:
                        entra, sale = simplexMax(Final)
                    xB[sale, 0] = entra
                    B[:, sale] = P[:, entra]
                    BInv = matrixBInv(B)
                    CB[sale] = c[entra]
                    Final = calcularMatrix(inicio, P, BInv, CB, b, c)
                    print(Final, "MAXIMIZAR")

            if (minimizar):
                while((min(Final[0, :-1]) > 0) or (min(Final[1:-1, restr + var]) < 0)):
                    if (min(Final[1:-1, restr + var]) < 0):
                        entra, sale = buscaDual(Final, restr, var)
                    else:
                        entra, sale = simplexMin(Final)

                    xB[sale, 0] = entra
                    print(xB, "xB")
                    B[:, sale] = P[:, entra]
                    BInv = matrixBInv(B)
                    CB[sale] = c[entra]
                    Final = calcularMatrix(inicio, P, BInv, CB, b, c)
                    print(Final, "MINIMIZAR")


            return Final, xB

        # Algoritmo Simlpex para 1Maximizar
        def simplexMax(Final):
            maxEntra = np.argmin(Final[0, :-1])
            dividir = []
            nocero = []
            for x in range(restr):
                if (Final[x + 1, maxEntra] > 0):
                    div = Final[x + 1, -2] / Final[x + 1, maxEntra]
                    dividir.append(div)
                else:
                    div = 0
                    dividir.append(div)
            for x in range(len(dividir)):
                if (dividir[x] > 0):
                    nocero.append(dividir[x])
            maxSale = np.where(dividir == np.min(nocero))[0][0]

            return maxEntra, maxSale

        # Algoritmo Simplex para Minimizar
        def simplexMin(Final):
            minEntra = np.argmax(Final[0, :-1])
            dividir = []
            nocero = []
            for x in range(restr):
                if (Final[x + 1, minEntra] > 0):
                    div = Final[x + 1, -2] / Final[x + 1, minEntra]
                    dividir.append(div)
                else:
                    div = 0
                    dividir.append(div)
            for x in range(len(dividir)):
                if (dividir[x] > 0):
                    nocero.append(dividir[x])
            minSale = np.where(dividir == np.min(nocero))[0][0]

            return minEntra, minSale

        # Produce la matriz aplicada al DUAL
        def buscaDual(Final, restr, var):
            # Buscar la variable a sacar
            salida = np.argmin(Final[1:-1, restr + var])
            print(salida, "Sale")
            dividir = []
            nocero = []
            # Buscar la variable a meter
            for x in range(restr + var):
                if (Final[1 + salida, x] < 0):
                    div = Final[0, x] / Final[1 + salida, x]
                    dividir.append(div)
                else:
                    div = 0
                    dividir.append(div)
            # Buscar en donde esta el minimo de la division sin ceros
            print(dividir, "DIVIDIR")
            for x in range(len(dividir)):
                if (dividir[x] != 0 and (x<var)):
                    nocero.append(dividir[x])
            if(nocero == []):
                for x in range(len(dividir)):
                    if (dividir[x] > 0):
                        nocero.append(dividir[x])
            print(nocero, "NO CERO")
            entrada = np.where(dividir == np.min(nocero))[0][0]
            print(entrada, "Entra")

            return entrada, salida

        # Crea la matrix P
        def matrixP(A, restr, var):
            (m, n) = A.shape
            table = np.zeros((m, m + n))
            table[:, :n] = A
            table[range(m), range(n, n + m)] = 1
            return table

        # Crea la matrix B
        def matrixB(P, var):
            return P.copy()[:, var:]


        # Crea la matriz invertida de B UTILIZANDO DESCOMPOSICION LU
        def matrixBInv(M):

            def descomp_LU(A):
                # Hace la descomposicion LU utilizando la factorizacion Doolittle

                L = np.zeros_like(A)
                U = np.zeros_like(A)
                N = np.size(A, 0)

                for k in range(N):
                    L[k, k] = 1
                    U[k, k] = (A[k, k] - np.dot(L[k, :k], U[:k, k])) / L[k, k]
                    for j in range(k + 1, N):
                        U[k, j] = (A[k, j] - np.dot(L[k, :k], U[:k, j])) / L[k, k]
                    for i in range(k + 1, N):
                        L[i, k] = (A[i, k] - np.dot(L[i, :k], U[:k, k])) / U[k, k]

                return L, U

            L,U = descomp_LU(M)
            #Regresa la matriz de la multiplicacion de las inversas de L y U
            return(np.matmul(np.linalg.inv(U),np.linalg.inv(L)))
            #return np.linalg.inv(M)




        # Crea la matrix de recursos b
        def matb(b):
            return b.reshape(-1, 1)

        # Crea la matriz CB
        def matCB(restr):
            return np.zeros((1, restr))[0]

        # Crea la matriz xB
        def matX(restr, var):
            xB = np.array(range(var, restr + var))
            xB = xB.reshape(-1, 1)
            return xB

        # Crea una matriz de ceros del tamaño correcto
        def matrixInicio(restr, var):
            inicio = np.zeros((restr + 2, restr + var + 2))
            return inicio

        # Llena la matriz inicial con las multiplicaciones de matrices
        def calcularMatrix(inicio, P, BInv, CB, b, c):
            inicio[0, :restr + var] = np.matmul(CB, np.matmul(BInv, P)) - c
            inicio[0, restr + var] = np.matmul(CB, np.matmul(BInv, b))
            inicio[1:-1, :-2] = np.matmul(BInv, P)
            inicio[1:-1, restr + var] = np.matmul(BInv, b).T
            return inicio

        try:
            resultado, xB = Inicializar(A, b, self.totalrestr, self.totalvar, mm)
            finalString = ""

            for x in range(restr):
                if (xB[x, 0] < var):
                    finalString += ("X" + str(xB[x, 0] + 1) + ": " + str(resultado[x + 1, -2]) + "\n")
            if (maximizar):
                finalString += "MAX: " + str(resultado[0, -2])
            if (minimizar):
                finalString += "MIN: " + str(resultado[0, -2])
            self.varmostrar.set(finalString)
        except:
            self.varmostrar.set("No hay solucion")

        return 0
'''
#Ejemplo de Examen
mm="Minimizar"

if(mm=="Maximizar"):
    maximizar = True
if(mm=="Minimizar"):
    minimizar = True
var = 2
restr = 3
c = np.array([180, 160, 0, 0, 0])
A = np.array([
    [-6, -1],
    [-3, -1],
    [-4, -6],
])
b = np.array([-12, -8, -24])
#resultado, xB = Inicializar(A,b, restr, var,mm)
print(xB, "MATRIX FINAL")
'''


def main():
    mi_app = Application()
    return 0

if __name__ == "__main__":
    main()