import tensorflow as tf
import pandas as pd
import numpy as np
import oracledb as odb
import os

from tensorflow.keras.models import load_model

# GPU 디바이스 설정
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # GPU 메모리 증분 가능하도록 설정
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        # 첫 번째 GPU만 사용하도록 설정 (선택 사항)
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

    except RuntimeError as e:
        print(e)

# DATABASE 연결을 위한 설정
LOCATION = r"D:\instantclient_21_11"
os.environ["PATH"] = LOCATION + ";" + os.environ["PATH"]  #환경변수 등록

# DATABASE 연결 부분
connect = odb.connect(user="BLRT2_STATUS", password="!tkdxo@$", dsn="blrt.db.nemosys.ai:15210/NEMOPDB1")
cursor = connect.cursor()

# 1. 모델 불러오기
model = load_model('./model/best_cnn_reg_model.h5')
print(model.summary())

# 모든 타이어의 ID 값을 SELECT
df=pd.read_sql('SELECT DISTINCT TM.TIRE_ID AS "타이어ID" FROM TIRE_MASTER TM ORDER BY TM.TIRE_ID', con = connect)

# DB에 있는 모든 타이어 들의 상태를 UPDATE OR INSERT
for i in range(len(df)):
	t_id = df.iloc[i].item()

	# 가장 최근 날짜 SELECT
	max_day = pd.read_sql('SELECT TO_CHAR(MAX(SMI.MEASURE_DTTM), \'YYYY-MM-DD\') AS "수신일시" \
	                            FROM SENSOR_MEASURE_INFO SMI\
	                                   JOIN SENSOR_MASTER SM ON SMI.SENSOR_ID = SM.SENSOR_ID\
	                                   JOIN SENSOR_LOCATION_MAPP SLM ON SM.SENSOR_ID = SLM.SENSOR_ID\
	                                   JOIN TIRE_MASTER TM ON TM.TIRE_ID = SLM.TIRE_ID\
	                                   JOIN TIRE_LOCATION_MAPP TLM ON TM.TIRE_ID = TLM.TIRE_ID\
	                            WHERE 1 = 1\
	                                   AND TM.TIRE_ID = \'%s\'' % t_id, con=connect)

	if max_day['수신일시'].isnull().all():
		continue
	else:
		date_value = pd.to_datetime(max_day.loc[0, '수신일시'])

	# 하나의 타이어의 온도 공기압 DATA SELECT
	yes_tire_info = pd.read_sql('SELECT SMI.MEASURE_DTTM AS "수신일시"\
	                                     , SMI.TEMPER AS "온도"\
	                                     , SMI.APRS AS "공기압"\
	                                  FROM SENSOR_MEASURE_INFO SMI\
	                                       JOIN SENSOR_MASTER SM ON SMI.SENSOR_ID = SM.SENSOR_ID\
	                                       JOIN SENSOR_LOCATION_MAPP SLM ON SM.SENSOR_ID = SLM.SENSOR_ID\
	                                       JOIN TIRE_MASTER TM ON TM.TIRE_ID = SLM.TIRE_ID\
	                                       JOIN TIRE_LOCATION_MAPP TLM ON TM.TIRE_ID = TLM.TIRE_ID\
	                                 WHERE 1 = 1\
	                                   AND TM.TIRE_ID = \'%s\'\
	                                   AND MEASURE_DTTM >= TO_DATE(\'%s\', \'YYYY-MM-DD\') -2\
	                                   AND MEASURE_DTTM < TO_DATE(\'%s\', \'YYYY-MM-DD\') -1\
	                                 ORDER BY SMI.MEASURE_DTTM DESC' % (
	t_id, date_value.strftime('%Y-%m-%d'), date_value.strftime('%Y-%m-%d')), con=connect)

	# 하나의 타이어의 온도 공기압 DATA SELECT
	tod_tire_info = pd.read_sql('SELECT SMI.MEASURE_DTTM AS "수신일시"\
	                                     , SMI.TEMPER AS "온도"\
	                                     , SMI.APRS AS "공기압"\
	                                  FROM SENSOR_MEASURE_INFO SMI\
	                                       JOIN SENSOR_MASTER SM ON SMI.SENSOR_ID = SM.SENSOR_ID\
	                                       JOIN SENSOR_LOCATION_MAPP SLM ON SM.SENSOR_ID = SLM.SENSOR_ID\
	                                       JOIN TIRE_MASTER TM ON TM.TIRE_ID = SLM.TIRE_ID\
	                                       JOIN TIRE_LOCATION_MAPP TLM ON TM.TIRE_ID = TLM.TIRE_ID\
	                                 WHERE 1 = 1\
	                                   AND TM.TIRE_ID = \'%s\'\
	                                   AND MEASURE_DTTM >= TO_DATE(\'%s\', \'YYYY-MM-DD\') -1\
	                                   AND MEASURE_DTTM < TO_DATE(\'%s\', \'YYYY-MM-DD\')\
	                                 ORDER BY SMI.MEASURE_DTTM DESC' % (
	t_id, date_value.strftime('%Y-%m-%d'), date_value.strftime('%Y-%m-%d')), con=connect)

	print(yes_tire_info.shape)
	print(tod_tire_info.shape)

	if yes_tire_info['온도'].isnull().all() or tod_tire_info['온도'].isnull().all():
		continue
	else:

		X_yes = yes_tire_info['온도'].values  # 온도 데이터
		y_yes = yes_tire_info['공기압'].values  # 공기압 데이터
		X_tod = tod_tire_info['온도'].values  # 온도 데이터
		y_tod = tod_tire_info['공기압'].values  # 공기압 데이터

		X_yes_sc = (X_yes - np.mean(X_yes)) / np.std(X_yes)
		X_tod_sc = (X_tod - np.mean(X_tod)) / np.std(X_tod)
		y_yes_sc = (y_yes - np.mean(y_yes)) / np.std(y_yes)
		y_tod_sc = (y_tod - np.mean(y_tod)) / np.std(y_tod)

		X_yes_re = np.reshape(X_yes_sc, (-1, 1))
		X_tod_re = np.reshape(X_tod_sc, (-1, 1))

		yes_pre = model.predict(X_yes_re)
		tod_pre = model.predict(X_tod_re)

		yes_orig = yes_pre * np.std(y_yes) + np.mean(y_yes)
		tod_orig = tod_pre * np.std(y_tod) + np.mean(y_tod)

		pre_dif = np.mean(tod_orig) - np.mean(yes_orig)

		print('평균 공기압 차이 : ', pre_dif)

		new_status = 0

		if pre_dif<-10 or pre_dif>10:
			new_status = 1

		# STATUS_SCORE_RECEIVE 테이블에 정보가 있는지 유효성 검사
		df_status_info = pd.read_sql("SELECT TIRE_ID FROM STATUS_SCORE_RECEIVE ssr WHERE TIRE_ID = '%s'" % t_id, con=connect)

		# STATUS_SCORE_RECEIVE SCORE INSERT OR UPDATE
		if len(df_status_info) == 0:
			insert_data = [t_id, new_status, 6]
			sql = "INSERT INTO STATUS_SCORE_RECEIVE (TIRE_ID, STATUS_SCORE, REG_USER_ID, REG_DTTM) VALUES (:1,:2,:3,SYSDATE)"
			cursor.execute(sql, insert_data)
		else:
			update_data = [new_status, 6, t_id]
			sql = "UPDATE STATUS_SCORE_RECEIVE SET STATUS_SCORE = :1, MOD_USER_ID = :2, MOD_DTTM = SYSDATE WHERE TIRE_ID = :3"
			cursor.execute(sql, update_data)

cursor.close()
connect.commit()
connect.close()