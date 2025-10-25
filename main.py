# Imports

import sys
from PyQt5.QtWidgets import QApplication
from gui import EdgeExplorer

def main():
    app = QApplication(sys.argv) # creates application instance
    win = EdgeExplorer(display_size=480) # creates main window
    
    win.show()
    sys.exit(app.exec_()) # starts event loop

# ensures main only runs when the script is directly executed
if __name__ == "__main__":
    main()
