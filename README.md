# AI-Project

## Overview
AI-Project is a machine learning-based application for credit card approval prediction. It utilizes a RandomForestClassifier to evaluate applicant data and determine approval outcomes.

## Project Structure
```
ai_project
│-- model/              # Stores trained models and scaler
│-- main.py             # AI model training and saving
│-- app.py              # Flask application for serving predictions
│-- requirements.txt    # Required dependencies
│-- README.md           # Project documentation
```

## Setup Instructions
### 1. Clone the Repository
```bash
git clone git@github.com:AmeyPacharkar1896/ai_project.git
cd ai_project
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run AI Model Training
Execute the following command to train and save the model:
```bash
python main.py
```
This will preprocess the data, train the model, and save it in the `model/` directory.

### 4. Start the Flask API
Once the model is trained and saved, run the Flask API:
```bash
python app.py
```

## API Usage
### Endpoint: Predict Credit Approval
**URL:** `http://127.0.0.1:5000/predict`
**Method:** `POST`
**Content-Type:** `application/json`

#### Example Request
```json
{
    "feature1": value1,
    "feature2": value2,
    ... (Total 69 features)
}
```

#### Example Response
```json
{
    "approval": "Approved"  // or "Not Approved"
}
```

## Notes
- Ensure `main.py` is executed before running `app.py`.
- Adjust hyperparameters in `main.py` for better performance.
- Use Postman or cURL to test API predictions.

## Contributors
- Amey Pacharkar

## License
This project is open-source and available under the MIT License.

