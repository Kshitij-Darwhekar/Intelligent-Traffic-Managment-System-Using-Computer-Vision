# Intelligent Traffic Management System using Computer Vision

Published Paper : https://ieeexplore.ieee.org/abstract/document/10009105 

Team : [Kshitij Darwhekar](https://www.linkedin.com/in/kshitij-darwhekar-b15a33191/),[Rushikesh Bawkar](https://www.linkedin.com/in/rushikesh-bawkar-874a931ba/) , [Sankalp Ghodke](https://www.linkedin.com/in/sankalpghodke/) , [Amey Patil](https://www.linkedin.com/in/amey-patil-7200b4213/)

## Table of Contents
1. [Introduction](#introduction)
2. [Features](#features)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Usage](#usage)


## Introduction
The Intelligent Traffic Management System is a computer vision-based application designed to monitor and manage traffic in real-time. The system utilizes computer vision algorithms and deep learning models to process live or recorded traffic videos, detect vehicles, analyze traffic flow, and generate intelligent traffic insights. This project aims to improve overall traffic management, reduce congestion, and enhance road safety by providing valuable data to traffic authorities and planners.

## Features
- Real-time vehicle detection: The system can detect vehicles in a live traffic feed and track their movements.
- Traffic flow analysis: Analyze the traffic flow and congestion patterns at different times of the day.
- Vehicle counting: Count the number of vehicles passing through specific road segments.
- Speed detection: Measure the speed of vehicles to identify potential speed violators.
- Incident detection: Detect and alert on unusual events such as accidents or road blockages.


## Requirements
- Python 3.x
- OpenCV
- YOLO Algorithm
- Numpy
- Pandas
- Matplotlib (for visualization)

## Installation
1. Clone this repository to your local machine.
2. Install the required dependencies using `pip` or `conda`:
   ```bash
   pip install opencv-python numpy pandas matplotlib
   ```
   or
   ```bash
   conda install opencv  numpy pandas matplotlib
   ```
3. Download the pre-trained model weights (if required) and place them in the appropriate directories.

## Usage
1. Open a terminal or command prompt and navigate to the project directory.
2. Run the traffic management system application:
   ```bash
   python traffic_management_system.py
   ```
3. The application will start processing the video feed and display the results, including vehicle detection, traffic flow analysis, and other insights.


## Contributing
Contributions to this project are welcome! If you find any issues or want to enhance the system's capabilities, feel free to open an issue or submit a pull request.
