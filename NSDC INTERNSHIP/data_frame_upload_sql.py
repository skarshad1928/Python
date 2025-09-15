from sqlalchemy import create_engine
import pandas as pd

# MySQL connection string
engine = create_engine("mysql+pymysql://root:NewPassword@localhost:3306/telecomdb")

# Load Excel files into DataFrames
data_frame = pd.read_excel(r"C:\Users\Shaik Arshad\OneDrive\Documents\GitHub\Python\NSDC INTERNSHIP\Final_Complete_cleaned.xlsx")
data_frame1 = pd.read_excel(r"C:\Users\Shaik Arshad\OneDrive\Documents\GitHub\Python\NSDC INTERNSHIP\Complete_location.xlsx")

# Push DataFrames to MySQL
data_frame.to_sql("Complete_dataset", con=engine, if_exists="replace", index=False)
data_frame1.to_sql("Complete_Location", con=engine, if_exists="replace", index=False)

print("Data uploaded successfully ")
