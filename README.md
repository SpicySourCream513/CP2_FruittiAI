Model Setup:
1) YOLOv8N -  backend/models/weights/best_fruit.pt
2) CNN RESNET-50 Apple Variety Model - backend/models/custom/apple_type.h5 
3) CNN RESNET-50 Apple Quality Assessment Model - backend/models/custom/apple_ripeness.keras

Backend Setup:
1.	Create Python virtual environment	
python -m venv apple_grading_env

2.	Activate the environment
apple_grading_env\Scripts\activate

3.	Install dependencies
pip install -r requirements.txt

4.	Start the backend server
python main.app

Frontend Setup:
1.	Install React
npm install

2.	Start the frontend server
npm start dev
