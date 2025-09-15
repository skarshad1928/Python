from sqlalchemy import create_engine,text
import pandas as pd

# MySQL connection string
engine = create_engine("mysql+pymysql://root:NewPassword@localhost:3306/telecomdb")
with engine.connect() as connection:
    result=connection.execute(text("select * from complete_dataset cd inner join complete_location cl on cd.LID=cl.LID;"))
    rows=result.fetchall()
df=pd.DataFrame(rows,columns=result.keys())
print(df)
df.to_excel("comlete_dataset.xlsx",index=False)
