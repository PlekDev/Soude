"""
run.py — Neuro-Lock launcher
Crea QApplication ANTES de importar app.py para evitar el error
'QWidget: Must construct a QApplication before a QWidget'
"""
import sys
from PyQt6.QtWidgets import QApplication

# QApplication debe existir antes de que cualquier widget se cree.
# app.py tiene codigo a nivel global que puede tocar Qt al importarse,
# por eso lo importamos DESPUES de crear QApplication.
app = QApplication(sys.argv)

from app import main  # noqa: E402  (import intencional despues de QApplication)
main()