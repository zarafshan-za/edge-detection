# main.py
# Launcher for the Edge Detection Explorer application.

import sys
from PyQt5.QtWidgets import QApplication
from gui import EdgeExplorer

def main():
    app = QApplication(sys.argv)
    win = EdgeExplorer(display_size=480)
    
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
