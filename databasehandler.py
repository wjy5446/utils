import pymysql

import pandas as pd

class DataBaseHandler():
    def __init__(self, db_type='mysql'):
        self.db_type = db_type

    # mysql
    def set_mysql(self, dict_info):
        self.dict_info = dict_info

        self.db_mysql = pymysql.connect(host=dict_info.get('host'),
                             user=dict_info.get('user'),
                             passwd=dict_info.get('pw'),
                             port=dict_info.get('port'),
                             db=dict_info.get('db'),
                             charset=dict_info.get('charset'))

        self.db_mysql.autocommit(True)
        self.cursor_mysql = self.db_mysql.cursor(pymysql.cursors.DictCursor)

    def execute_mysql(self, *sql, **kwargs):
        try:
            self.cursor_mysql.execute(*sql)
        except Exception as e:
            print(e)
            self.set_mysql(self.dict_info)

        if 'select' in sql[0].lower():
            if kwargs.get('one'):
                result = self.cursor_mysql.fetchone()
            else:
                result = self.cursor_mysql.fetchall()

            return result
        else:
            return None

    def execute_mysql_dataframe(self, sql):
        df_tmp = pd.read_sql_query(sql, self.db_mysql)
        return df_tmp

if __name__ == '__main__':
    dbh = DataBaseHandler()

    dict_db_info = {
        'host': 'db.ds.mycelebs.com',
        'user': 'celebDev',
        'pw': 'epdlxjtkdldjstm!',
        'port': 3306,
        'db': 'movie_eng',
        'charset': 'utf8'
    }

    #dbh.set_mysql(dict_db_info)

    sql = 'select * from a_imdb_basics_movie_review limit 10'
    row = dbh.execute_mysql(sql)
    print(row[0])