import mysql.connector
import datetime


class face_mysql:
    def __init__(self):
        pass

    def conn_mysql(self):
        db = mysql.connector.connect \
            (user='root',
             password='lhp19980901',
             host='127.0.0.1',
             database='Face_json')
        return db

    def create_table(self):
        db = self.conn_mysql()
        cursor = db.cursor()

        use_db_sql = 'Face_json;'
        sql =  """
        CREATE TABLE IF NOT EXISTS `face_json` (
        `id` int(32) NOT NULL AUTO_INCREMENT COMMENT 'id自增',
        `ugroup` varchar(255) DEFAULT NULL COMMENT 'users groups',
        `uid` varchar(64) DEFAULT NULL COMMENT 'The picture of user',
        `json` text COMMENT 'The vector of human face',
        `pic_name` varchar(255) DEFAULT NULL COMMENT 'the name of pic',
        `date` datetime DEFAULT NULL COMMENT 'insert_time',
        `state` tinyint(1) DEFAULT NULL,
        PRIMARY KEY (`id`)
        ) ENGINE=InnoDB AUTO_INCREMENT=1 DEFAULT CHARSET=utf8;
        """

        try:
            cursor.seecute(sql)
            print('Create table successfully!')
        except:
            print('Error: unable o create table ')

    def insert_facejson(self, pic_name, pic_json, uid, ugroup):
        db = self.conn_mysql()
        cursor = db.cursor()
        dt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sql = "insert into face_json(json,pic_name,date,state,uid,ugroup) values('%s' ,'%s','%s','%d','%s','%s') ;" % (
            pic_json, pic_name, dt, 1, uid, ugroup)
        #print("sql=",sql)
        try:
            cursor.execute(sql)
            lastid = int(cursor.lastrowid)
            db.commit()
        except:
            # Rollback in case there is any error
            db.rollback()
        db.close()
        return lastid

    def findall_facejson(self, ugroup):
        db = self.conn_mysql()
        cursor = db.cursor()

        sql = "select * from face_json where state=1 and ugroup= '%s' ;" % (ugroup)
        try:
            cursor.execute(sql)
            results = cursor.fetchall()
            return results
        except:
            print("Error:unable to fecth data")
        db.close()