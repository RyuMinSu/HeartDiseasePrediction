import pandas as pd
import pymysql
from sqlalchemy import create_engine

def creatTb(dfpath, user, pw, host, dbName, tbName):
    df = pd.read_csv(dfpath)
    dbConnpath = f"mysql+pymysql://{user}:{pw}@{host}/{dbName}"
    dbConn = create_engine(dbConnpath)
    conn = dbConn.connect()
    df.to_sql(name=tbName, con=conn, if_exists="fail", index=False)

def readDb(host, user, pw, dbName, rsql):
    conn = pymysql.connect(host=host, user=user, passwd=str(pw), db=dbName, charset="utf8")
    cur = conn.cursor()
    df = pd.read_sql(rsql, con=conn)
    return df