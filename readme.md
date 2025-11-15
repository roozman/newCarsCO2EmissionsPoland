## Problem Description
Vehicle emissions, particularly CO2, are a significant contributor to global climate change. Accurately predicting CO2 emissions based on vehicle specifications (such as engine size, fuel type, and weight) is crucial for regulatory compliance, environmental impact assessments, and consumer awareness. Traditional rule-based approaches often fail to capture the complex, non-linear relationships between vehicle features and emissions. Machine learning excels in this context by leveraging large datasets to identify patterns, optimize predictions, and adapt to new data—making it an ideal solution for modeling and forecasting CO2 emissions with high accuracy and scalability.


### Why It Matters
Transportation is a major contributor to greenhouse gas emissions. Accurate CO2 predictions enable:
- **Policy compliance**: Governments can monitor and enforce emission targets.
- **Consumer awareness**: Buyers can compare vehicles’ environmental impact.
- **Industry innovation**: Manufacturers can prioritize low-emission designs.

### Evaluation Metric
Since this is a **regression task** (predicting a continuous value), the model’s performance is evaluated using:
- **Root Mean Squared Error (RMSE)**: Measures prediction accuracy, penalizing larger errors more heavily.
- **R² Score**: Indicates the proportion of variance in CO2 emissions explained by the model, providing insight into overall fit and predictive power.



## Data Source
The dataset used in this project was sourced from [data.europa.eu](https://data.europa.eu/), a comprehensive portal for European open data. Given the massive size of the full dataset, I focused on a subset to keep the project manageable and relevant. Specifically, I filtered the data to include only vehicles registered in **Poland**, as this provided a balanced and representative sample for modeling CO2 emissions.

For convenience, you can download the pre-filtered dataset directly using this link:
 [Pre-filtered Poland CO2 Emissions Data (Proton Drive)](https://drive.proton.me/urls/7X386KENPM#VFw2Gi2EfEOs)

Alternatively, if you prefer to filter the data yourself, you can use the original source with the pre-selected filter:
 [CO2 Cars Data (EEA Europa)](https://co2cars.apps.eea.europa.eu/?source=%7B%22track_total_hits%22%3Atrue%2C%22query%22%3A%7B%22bool%22%3A%7B%22must%22%3A%7B%22bool%22%3A%7B%22must%22%3A%5B%7B%22range%22%3A%7B%22Ewltp__g_km_%22%3A%7B%22from%22%3A%2210%22%2C%22to%22%3A%22250%22%7D%7D%7D%2C%7B%22constant_score%22%3A%7B%22filter%22%3A%7B%22bool%22%3A%7B%22must%22%3A%5B%7B%22bool%22%3A%7B%22should%22%3A%5B%7B%22term%22%3A%7B%22year%22%3A2024%7D%7D%5D%7D%7D%2C%7B%22bool%22%3A%7B%22should%22%3A%5B%7B%22term%22%3A%7B%22scStatus%22%3A%22Provisional%22%7D%7D%5D%7D%7D%5D%7D%7D%7D%7D%5D%7D%7D%2C%22filter%22%3A%7B%22bool%22%3A%7B%22should%22%3A%5B%7B%22term%22%3A%7B%22MS%22%3A%22PL%22%7D%7D%5D%7D%7D%7D%7D%2C%22display_type%22%3A%22tabular%22%7D)

Once downloaded, place the dataset in the `/data` directory as `data.csv` to ensure the scripts run correctly.

## Exploratory Data Analysis (EDA)
The Jupyter Notebook [`EDA.ipynb`](EDA.ipynb) contains detailed **Exploratory Data Analysis (EDA)**, including:
- Data cleaning and preprocessing steps
- Feature importance analysis
- Visualizations of key trends and relationships

This notebook provides information into the data preparation process and insights that guided model selection.

### Exploratory Data Analysis (EDA) Summary

The EDA process helped us to get to know the dataset and ensured our preprocessing pipeline was on the right track. Our target variable is  **`ewltp_(g/km)` (CO2 emissions)**.

**Key Findings & Why They Matter**

The EDA confirmed that our modeling approach makes sense:

- **Feature Selection**: We dropped columns with over 90% missing values, leaving us with **8 key features** that actually matter for prediction.

- **Handling Missing Values & Scaling**:
  - Numerical features like **`mt` (WLTP test mass)**, **`ec_(cm3)` (engine capacity)**, and **`ep_(kw)` (engine power)** had only a few missing values, so we used **`SimpleImputer(strategy='median')`** to fill them in.
  - Since these features had very different scales, **`StandardScaler`** was applied to keep everything balanced.

- **Correlations**:
  - Features like **`ep_(kw)`**, **`ec_(cm3)`**, and **`m_(kg)` (mass)** showed a **strong positive correlation** with CO2 emissions—meaning they’re great predictors.

- **Encoding Strategy**:
  - **High-Cardinality Features**:
    - **`mk` (Make, 51 unique values)** and **`ech` (Engine Standard, 17 unique values)** were encoded using **`TargetEncoder`** to capture their relationship with emissions.
  - **Low-Cardinality Features**:
    - **`ft` (Fuel Type, 5 unique values)** and **`fm` (Fuel Mode, 4 unique values)** were encoded using **`OneHotEncoder`** for simplicity and clarity.


## Model Selection
The model selection process is documented in [`notebook.ipynb`](notebook.ipynb). Key steps include:

- **Baseline Model**: A simple `LinearRegression` was tested to establish a performance baseline. This helped verify whether the final model’s performance was due to its effectiveness or potential issues in data preprocessing.

- **Final Model**: Both **`LGBMRegressor`** and **`XGBoost`** were evaluated, with very close performance scores. However, **`LGBMRegressor`** was chosen as the final model due to its **faster training time** and **better handling of categorical variables**, which is critical given the dataset size (570,000+ rows). The model was fine-tuned using **`GridSearchCV`** for optimal hyperparameters.

## How to Run the Project

### Running Locally

1. **Clone the repository**
   ```bash
   git clone https://github.com/roozman/newCarsCO2EmissionsPoland.git
   cd newCarsCO2EmissionsPoland
2. **Setting up the environment**
   ```bash
   pip install -r requirements.txt
3. **Train the model**
   ```bash
   python main.py
The service will start at http://0.0.0.0:8000.

### Running with Docker

1. **Build the docker image**
   ```bash
   docker build -t co2-emissions .
2. **Run the container**
   ```bash
   docker run -p 8000:8000 co2-emissions
The service will start at http://0.0.0.0:8000.

## API Usage Example

The API provides a `/predict` endpoint to predict CO2 emissions based on vehicle specifications. You can easily test and interact with the API using FastAPI's built-in documentation.

### Using FastAPI Docs

1. **Start the Service**
   Make sure the service is running locally or via Docker:
   ```bash
   python main.py
2. **Access API docs**
   ```bash
   http://0.0.0.0:8000/docs
3. **Test the predict endpoint** <br>
You can edit the placeholder values for the model and then press **Execute** to test the model.   
4. **View the response** <br>
The API will return a JSON response with the predicted CO2 emissions, for example:
   ```bash
   {
   "predictions": 125.5
   }
## Project Structure

├── README.md <br>
├── data/       # Directory for dataset <br>
│   └── data.csv (Download instruction in readme) <br>
│   └── Table-definition.xlsx # Feature guide <br>
├── notebook.ipynb  # Model selection and evaluation<br>
├── train.py # Script to train and save the final model<br>
├── predict.py  # Script to load the model and serve predictions via FastAPI<br>
├── main.py # script to run the project (in place of serve.py) <br>
├── Dockerfile  # Docker configuration for containerization<br>
├── Model.bin # Exported model<br>
├──EDA.ipynb <br>
└── requirements.txt  <br>


