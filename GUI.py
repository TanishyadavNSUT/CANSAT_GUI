import sys
import os
import pandas as pd
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QFont, QColor, QPalette, QLinearGradient, QPixmap, QImage
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QWidget, QFrame, QGridLayout
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import datetime
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtNetwork import QNetworkRequest
from PyQt5.QtCore import QUrl
import cv2


class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure()
        super().__init__(self.fig)
        self.setParent(parent)
        self.axes = self.fig.add_subplot(111)

    def update_plot(self, x, y, title, xlabel, ylabel):
        self.axes.clear()
        self.axes.plot(x, y, label=title)
        self.axes.set_xlabel(xlabel, fontsize=10)
        self.axes.set_ylabel(ylabel, fontsize=10)
        self.axes.legend(fontsize=8)
        self.axes.grid(True, alpha=0.7)
        self.fig.tight_layout()
        self.draw()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("TEAM KALPANA : 2024-CANSAT-ASI-023")
        self.setGeometry(100, 100, 1200, 800)

        self.data = pd.DataFrame()
        self.data_index = 0
        self.time_data = []

        self.altitude = []
        self.pressure = []
        self.voltage = []
        self.gyro_r = []
        self.acc_r = []
        self.gnss_altitude = []

        self.load_data()
        self.init_ui()
        self.start_timer()

    def load_data(self):
        try:
            if os.path.exists("data.csv"):
                self.data = pd.read_csv("data.csv")
            else:
                print("Warning: 'data.csv' not found. Using empty data.")
        except Exception as e:
            print(f"Error loading CSV file: {e}")

    def init_ui(self):
        palette = QPalette()
        gradient = QLinearGradient(0, 0, 0, 1)
        gradient.setCoordinateMode(QLinearGradient.ObjectBoundingMode)
        gradient.setColorAt(0.0, QColor(47, 38, 95)) 
        gradient.setColorAt(1.0, QColor(60, 194, 255)) 
        palette.setBrush(QPalette.Window, gradient)
        self.setPalette(palette)

        self.tabs = QTabWidget()
        self.tabs.setFont(QFont("Arial", 12))
        self.setCentralWidget(self.tabs)
        self.tabs.setStyleSheet("""
    QTabBar::tab {
        height: 40px;  
        width: 240px;  
        font-size: 14px; 
        font-family: 'Segoe UI', Arial, sans-serif;
        font-weight: 600; 
        color: #004466; 
        background: #E7F6F8; 
        border: 1px solid #99C2CC;  
        border-radius: 6px;  
        padding: 6px;
        margin: 6px; 
        text-align: center;  
    }
    QTabBar::tab:hover {
        background: #CCE7EE; 
        color: #003344;  
        border: 1px solid #007C92; 
    }
    QTabBar::tab:selected {
        background: #007C92; 
        color: white;  
        border: 1px solid #005566; 
    }
    QTabBar {
        alignment: center;  
        margin: 5px;  
    }
""")


        self.init_tabs()
        self.init_header()
        self.init_footer()
        

    def init_tabs(self):
        self.telemetry_tab = QWidget()
        self.graph_tab = QWidget()
        self.location_tab = QWidget()
        self.telecast_tab = QWidget()

        # Add tabs to the TabWidget
        self.tabs.addTab(self.telemetry_tab, "Telemetry Data")
        self.tabs.addTab(self.graph_tab, "Graphs")
        self.tabs.addTab(self.location_tab, "Location and 3D Plotting")
        self.tabs.addTab(self.telecast_tab, "Live Telecast")

        self.init_graph_tab()
        self.init_telecast_tab()
    
    def init_telecast_tab(self):
        layout = QVBoxLayout()
        self.telecast_tab.setLayout(layout)

        self.video_label = QLabel("Loading Stream...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; color: white; font-size: 18px;")
        layout.addWidget(self.video_label)

        refresh_button = QPushButton("Refresh Stream")
        refresh_button.setFont(QFont("Arial", 14))
        refresh_button.setStyleSheet("""
            QPushButton {
                background-color: #4682B4; 
                color: white; 
                border-radius: 8px; 
                padding: 8px 16px; 
            }
            QPushButton:hover {
                background-color: #1E90FF;
            }
            QPushButton:pressed {
                background-color: #4169E1;
            }
        """)
        refresh_button.clicked.connect(self.start_video_stream)
        layout.addWidget(refresh_button)

        self.stream_url = "http://192.168.29.142:8080/video"
        self.cap = None
        self.timer = QTimer()

        self.start_video_stream()

    def start_video_stream(self):
        if self.cap:
            self.cap.release()
        self.timer.stop()
        self.cap = cv2.VideoCapture(self.stream_url)
        if not self.cap.isOpened():
            self.video_label.setText("Failed to connect to stream.")
            return

        self.timer.timeout.connect(self.update_video_frame)
        self.timer.start(30)  

    def update_video_frame(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                height, width, channels = frame.shape
                q_image = QImage(frame.data, width, height, width * channels, QImage.Format_RGB888)

                self.video_label.setPixmap(QPixmap.fromImage(q_image))
            else:
                self.video_label.setText("Stream interrupted. Reconnecting...")
                self.start_video_stream()
        else:
            self.video_label.setText("Stream disconnected. Reconnecting...")
            self.start_video_stream()

    def init_graph_tab(self):
        layout = QVBoxLayout()

        graph_frame = QFrame()
        graph_frame.setStyleSheet("background-color: #E6E6FA; border-radius: 10px; padding: 10px;")
        graph_layout = QGridLayout()
        graph_frame.setLayout(graph_layout)
        layout.addWidget(graph_frame)

        self.graph_tab.setLayout(layout)

        self.graphs = {
            "Altitude": PlotCanvas(self),
            "Pressure": PlotCanvas(self),
            "Voltage": PlotCanvas(self),
            "Gyro_R": PlotCanvas(self),
            "ACC_R": PlotCanvas(self),
            "GNSS Altitude": PlotCanvas(self),
        }

        titles = list(self.graphs.keys())
        for i, title in enumerate(titles):
            title_label = QLabel(f"{title} vs Time")
            title_label.setFont(QFont("Arial", 18, QFont.Bold))
            title_label.setAlignment(Qt.AlignCenter)
            graph_layout.addWidget(title_label, i // 3 * 2, i % 3)
            graph_layout.addWidget(self.graphs[title], i // 3 * 2 + 1, i % 3)

    def init_header(self):
        # Create header frame
        header_frame = QFrame()
        header_frame.setFixedHeight(140)
        header_frame.setStyleSheet("background-color: #2F265F; padding: 5px;") 

        header_layout = QHBoxLayout()

        software_state_layout = QVBoxLayout()
        software_state_label = QLabel("SOFTWARE STATE")
        software_state_label.setFont(QFont("Arial", 18, QFont.Bold)) 
        software_state_label.setStyleSheet("color: white; text-align: center;")
        
        launch_button = QPushButton("LAUNCH PAD")
        launch_button.setFont(QFont("Arial", 13, QFont.Bold))  
        launch_button.setFixedSize(150, 40) 
        launch_button.setStyleSheet("""
            QPushButton {
                background-color: #FF4136;
                color: white;
                border-radius: 20px;
                padding: 8px;
                border: 2px solid #FF5733;
            }
            QPushButton:hover {
                background-color: #FF5733;
                border-color: #FF6F61;
            }
            QPushButton:pressed {
                background-color: #C70039;
            }
        """)
        software_state_layout.addWidget(software_state_label, alignment=Qt.AlignCenter)
        software_state_layout.addWidget(launch_button, alignment=Qt.AlignCenter)

        team_logo_layout = QVBoxLayout()
        team_label = QLabel("TEAM KALPANA : 2024-CANSAT-ASI-023")
        team_label.setFont(QFont("Arial", 20, QFont.Bold))  
        team_label.setStyleSheet("color: white; text-align: center;")
        logo_label = QLabel()
        pixmap = QPixmap("logo.png")  
        pixmap = pixmap.scaled(70, 70, Qt.KeepAspectRatio, Qt.SmoothTransformation)  
        logo_label.setPixmap(pixmap)
        logo_label.setAlignment(Qt.AlignCenter)
        team_logo_layout.addWidget(team_label, alignment=Qt.AlignCenter)
        team_logo_layout.addWidget(logo_label, alignment=Qt.AlignCenter)

        time_layout = QVBoxLayout()
        time_heading = QLabel("TIME")
        time_heading.setFont(QFont("Arial", 16, QFont.Bold)) 
        time_heading.setStyleSheet("color: white; text-align: center;")
        self.time_label = QPushButton("00:00:00")
        self.time_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.time_label.setFixedSize(150, 40)  
        self.time_label.setStyleSheet("""
            QPushButton {
                background-color: #1E90FF;
                color: white;
                border-radius: 20px;
                border: 2px solid #4682B4;
            }
            QPushButton:hover {
                background-color: #4682B4;
            }
            QPushButton:pressed {
                background-color: #4169E1;
            }
        """)
        time_layout.addWidget(time_heading, alignment=Qt.AlignCenter)
        time_layout.addWidget(self.time_label, alignment=Qt.AlignCenter)

        packet_layout = QVBoxLayout()
        packet_heading = QLabel("PACKET COUNT")
        packet_heading.setFont(QFont("Arial", 16, QFont.Bold)) 
        packet_heading.setStyleSheet("color: white; text-align: center;")
        self.packet_label = QPushButton("0")
        self.packet_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.packet_label.setFixedSize(150, 40)  
        self.packet_label.setStyleSheet("""
            QPushButton {
                background-color: #32CD32;
                color: white;
                border-radius: 20px;
                border: 2px solid #228B22;
            }
            QPushButton:hover {
                background-color: #228B22;
            }
            QPushButton:pressed {
                background-color: #006400;
            }
        """)
        packet_layout.addWidget(packet_heading, alignment=Qt.AlignCenter)
        packet_layout.addWidget(self.packet_label, alignment=Qt.AlignCenter)

        header_layout.addLayout(software_state_layout)
        header_layout.addStretch()
        header_layout.addLayout(team_logo_layout)
        header_layout.addStretch()
        header_layout.addLayout(time_layout)
        header_layout.addLayout(packet_layout)

        header_frame.setLayout(header_layout)

        main_layout = QVBoxLayout()
        main_layout.addWidget(header_frame)
        main_layout.addWidget(self.tabs)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        self.start_clock()




    def start_clock(self):
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_time)
        self.timer.start(1000)

    def update_time(self):
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        self.time_label.setText(f"TIME: {current_time}")


    def init_footer(self):
        footer_frame = QFrame()
        footer_frame.setFixedHeight(100)
        footer_frame.setStyleSheet("""
            background-color: #EAF3F8;  
            border-top: 2px solid #B0D4E3; 
        """)

        footer_layout = QHBoxLayout()

        buttons = [
            "BOOT", "Set Time", "Calibrate", "ON / OFF",
            "CX", "SIM Enable", "SIM Activate", "SIM Disable",
        ]

        for text in buttons:
            button = QPushButton(text)
            button.setFont(QFont("Segoe UI", 12, QFont.Bold))  
            button.setStyleSheet("""
                QPushButton {
                    background-color: #D1E7F3; 
                    color: #004466; 
                    border: 1px solid #99C2CC;  
                    border-radius: 8px; 
                    padding: 8px 16px; 
                    margin: 4px;  
                }
                QPushButton:hover {
                    background-color: #B0D4E3; 
                    color: #003344;
                    border: 1px solid #007C92; 
                }
                QPushButton:pressed {
                    background-color: #007C92; 
                    color: white; 
                }
            """)
            button.clicked.connect(lambda _, b=text: print(b))
            footer_layout.addWidget(button)

        footer_frame.setLayout(footer_layout)

        central_layout = self.centralWidget().layout()
        central_layout.addWidget(footer_frame)



    def start_timer(self):
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_graphs)
        self.timer.start(1000)

    def update_graphs(self):
        if not self.data.empty and self.data_index < len(self.data):
            try:
                self.time_data.append(self.data_index)
                self.altitude.append(self.data["ALTITUDE"].iloc[self.data_index])
                self.pressure.append(self.data["PRESSURE"].iloc[self.data_index])
                self.voltage.append(self.data["VOLTAGE"].iloc[self.data_index])
                self.gyro_r.append(self.data["GYRO_R"].iloc[self.data_index])
                self.acc_r.append(self.data["ACC_R"].iloc[self.data_index])
                self.gnss_altitude.append(self.data["GNSS_ALTITUDE"].iloc[self.data_index])

                datasets = {
                    "Altitude": (self.altitude, "Altitude"),
                    "Pressure": (self.pressure, "Pressure"),
                    "Voltage": (self.voltage, "Voltage"),
                    "Gyro_R": (self.gyro_r, "Gyro_R"),
                    "ACC_R": (self.acc_r, "ACC_R"),
                    "GNSS Altitude": (self.gnss_altitude, "GNSS Altitude"),
                }

                for title, (values, ylabel) in datasets.items():
                    self.graphs[title].update_plot(
                        self.time_data, values, title, "Time", ylabel
                    )

                self.data_index += 1

            except KeyError as e:
                print(f"Missing column in data: {e}")
            except Exception as e:
                print(f"Error updating graphs: {e}")
    


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
