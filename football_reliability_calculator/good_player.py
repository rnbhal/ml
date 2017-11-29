import math
import sqlite3
import numpy as np
import pandas as pd
import sqlalchemy
from numpy.random import random
from sqlalchemy import create_engine
import seaborn as sns

conn = sqlite3.connect('database.sqlite')
c = conn.cursor()
import datetime
def get_zodiac_of_date(date):
    zodiacs = [(120, 'Cap'), (218, 'Aqu'), (320, 'Pis'), (420, 'Ari'), (521, 'Tau'),
           (621, 'Gem'), (722, 'Can'), (823, 'Leo'), (923, 'Vir'), (1023, 'Lib'),
           (1122, 'Sco'), (1222, 'Sag'), (1231, 'Cap')]
    date_number = int("".join((str(date.month), '%02d' % date.day)))
    for z in zodiacs:
        if date_number <= z[0]:
            return z[1]
def get_zodiac_for_football_players(x):
    date  =  x.split(" ")[0]
    date = datetime.datetime.strptime(date, "%Y-%m-%d").date()
    return get_zodiac_of_date(date)
def get_age_for_football_players(x):
    date  =  x.split(" ")[0]
    today = datetime.datetime.strptime("2015-01-01", "%Y-%m-%d").date()
    born = datetime.datetime.strptime(date, "%Y-%m-%d").date()
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))
def get_overall_rating(x):
    global c
    all_rating = c.execute("""SELECT overall_rating FROM Player_Stats WHERE player_api_id = '%d' """ % (x)).fetchall()
    all_rating = np.array(all_rating,dtype=np.float)[:,0]
    mean_rating = np.nanmean(all_rating)
    return mean_rating
def get_current_team_and_country(x):
    global c
    all_rating = c.execute("""SELECT overall_rating FROM Player_Stats WHERE player_api_id = '%d' """ % (x)).fetchall()
    all_rating = np.array(all_rating,dtype=np.float)[:,0]
    rating = np.nanmean(all_rating)
    if (rating>1): 
        all_football_nums = reversed(range(1,12))
        for num in all_football_nums:
            all_team_id = c.execute("""SELECT home_team_api_id, country_id FROM Match WHERE home_player_%d = '%d'""" % (num,x)).fetchall()
            if len(all_team_id) > 0:
                number_unique_teams = len(np.unique(np.array(all_team_id)[:,0]))
                last_team_id = all_team_id[-1][0]
                last_country_id = all_team_id[-1][1]
                last_country = c.execute("""SELECT name FROM Country WHERE id = '%d'""" % (last_country_id)).fetchall()[0][0]
                last_team = c.execute("""SELECT team_long_name FROM Team WHERE team_api_id = '%d'""" % (last_team_id)).fetchall()[0][0]
                return last_team, last_country, number_unique_teams
    return None, None, 0
def get_position(x):
    global c
    all_rating = c.execute("""SELECT overall_rating FROM Player_Stats WHERE player_api_id = '%d' """ % (x)).fetchall()
    all_rating = np.array(all_rating,dtype=np.float)[:,0]
    rating = np.nanmean(all_rating)
    if (rating>1): 
        all_football_nums = reversed(range(1,12))
        for num in all_football_nums:
            all_y_coord = c.execute("""SELECT home_player_Y%d FROM Match WHERE home_player_%d = '%d'""" % (num,num,x)).fetchall()
            if len(all_y_coord) > 0:
                Y = np.array(all_y_coord,dtype=np.float)
                mean_y = np.nanmean(Y)
                if (mean_y >= 10.0):
                    return "for"
                elif (mean_y > 5):
                    return "mid"
                elif (mean_y > 1):
                    return "def"
                elif (mean_y == 1.0):
                    return "gk"
    return None
with sqlite3.connect('database.sqlite') as con:
    sql = "SELECT * FROM Player"
    max_players_to_analyze = 10000
    players_data = pd.read_sql_query(sql, con)
    players_data = players_data.iloc[0:max_players_to_analyze]
    players_data["zodiac"] = np.vectorize(get_zodiac_for_football_players)(players_data["birthday"])
    players_data["rating"] = np.vectorize(get_overall_rating)(players_data["player_api_id"])
    players_data["age"] = np.vectorize(get_age_for_football_players)(players_data["birthday"])
    players_data["team"], players_data["country"], players_data["num_uniq_team"] = np.vectorize(get_current_team_and_country)(players_data["player_api_id"])
    players_data["position"] = np.vectorize(get_position)(players_data["player_api_id"])
    print(players_data)

positions = players_data["position"]
x_data = players_data[["rating", "height", "weight"]]
x_data["age"] = np.vectorize(get_age_for_football_players)(players_data["birthday"])
x_data.head()

def get_potential(x):
    global c
    all_rating = c.execute("""SELECT potential FROM Player_Stats WHERE player_api_id = '%d' """ % (x)).fetchall()
    all_rating = np.array(all_rating,dtype=np.float)[:,0]
    mean_rating = np.nanmean(all_rating)
    return mean_rating
def get_preferred_foot(x):
    global c
    all_rating = c.execute("""SELECT preferred_foot FROM Player_Stats WHERE player_api_id = '%d' """ % (x)).fetchall()
    if all_rating[0][0] == "right":
        return 0.0
    else:
        return 1.0
    return float("nan")
def get_attacking_work_rate(x):
    global c
    all_rating = c.execute("""SELECT attacking_work_rate FROM Player_Stats WHERE player_api_id = '%d' """ % (x)).fetchall()
    if all_rating[0][0] == "high":
        return 2.0
    if all_rating[0][0] == "medium":
        return 1.0
    if all_rating[0][0] == "low":
        return 0.0
    return float("nan")
def get_defensive_work_rate(x):
    global c
    all_rating = c.execute("""SELECT defensive_work_rate FROM Player_Stats WHERE player_api_id = '%d' """ % (x)).fetchall()
    if all_rating[0][0] == "high":
        return 2.0
    if all_rating[0][0] == "medium":
        return 1.0
    if all_rating[0][0] == "low":
        return 0.0
    return float("nan")
def get_anyone_statistic(x, col_name):
    global c
    all_rating = c.execute("""SELECT %s FROM Player_Stats WHERE player_api_id = '%d' """ % (col_name, x)).fetchall()
    all_rating = np.array(all_rating,dtype=np.float)[:,0]
    mean_rating = np.nanmean(all_rating)
    return mean_rating

train_x_data = x_data
train_x_data["potential"] = np.vectorize(get_potential)(players_data["player_api_id"])
train_x_data["defensive_work_rate"] = np.vectorize(get_defensive_work_rate)(players_data["player_api_id"])
train_x_data["attacking_work_rate"] = np.vectorize(get_attacking_work_rate)(players_data["player_api_id"])
train_x_data["preferred_foot"] = np.vectorize(get_attacking_work_rate)(players_data["player_api_id"])
all_columns = c.execute('PRAGMA TABLE_INFO(Player_stats)').fetchall()
for i in all_columns:
    if i[0] > 8:
        train_x_data[i[1]] = np.vectorize(get_anyone_statistic)(players_data["player_api_id"], i[1])
train_x_data.head()

new_train_x = train_x_data.fillna(train_x_data.median()).values
def pos_to_num(x):
    if x =="for":
        return 4
    elif x == "mid":
        return 3
    elif x == "def":
        return "2"
    elif x == "gk":
        return 1
positions = pd.DataFrame(positions)
positions["nums"] = np.vectorize(pos_to_num)(positions)
y_train = positions.fillna(positions.median())
y_train.fillna(value=np.nan, inplace=True)
temp = pd.to_numeric(y_train["nums"], errors='coerce')
temp = temp.fillna(temp.median())
y_train = temp.values
x_train = new_train_x


from sklearn.preprocessing import normalize
x_data = normalize(x_train)
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
x_data, y_train, test_size=0.25, random_state=42)

new_train_x = train_x_data.fillna(train_x_data.median()).values
def pos_to_num(x):
    if x =="for":
        return 4
    elif x == "mid":
        return 3
    elif x == "def":
        return "2"
    elif x == "gk":
        return 1
positions = pd.DataFrame(positions)
positions["nums"] = np.vectorize(pos_to_num)(positions)
y_train = positions.fillna(positions.median())
y_train.fillna(value=np.nan, inplace=True)
temp = pd.to_numeric(y_train["nums"], errors='coerce')
temp = temp.fillna(temp.median())
y_train = temp.values
x_train = new_train_x
x_data = normalize(x_train)
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
x_data, y_train, test_size=0.25, random_state=42)

players_data["preferred_foot"] = np.vectorize(get_preferred_foot)(players_data["player_api_id"])
def zodiac_to_num(x):
    zodiacs = ['Cap', 'Aqu', 'Pis', 'Ari', 'Tau',
           'Gem', 'Can', 'Leo', 'Vir', 'Lib',
           'Sco', 'Sag', 'Cap']
    return zodiacs.index(x)
def position_to_num(x):
    positions = ['gk', 'def', 'mid', 'for']
    if x in positions:
        return 1.0*positions.index(x)
    else:
        return float("nan")
    
data_for_prediction = players_data[["height", "weight", "zodiac", "age", "preferred_foot", "position", "rating"]]
data_for_prediction["zodiac"] = np.vectorize(zodiac_to_num)(data_for_prediction["zodiac"])
data_for_prediction["position"] = np.vectorize(position_to_num)(data_for_prediction["position"])
data_for_prediction = data_for_prediction.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
rating = data_for_prediction["rating"]
good_players = ((rating > 70)*1).values
data_for_prediction = data_for_prediction.drop(["rating"], axis=1).values
print(data_for_prediction)
# data_for_prediction.head()

from sklearn import tree
from sklearn.cross_validation import train_test_split
clf = tree.DecisionTreeClassifier(max_depth = 5, min_samples_leaf = 10)
x_train, x_test, y_train, y_test = train_test_split(
data_for_prediction, good_players, test_size=0.25, random_state=42)
clf = clf.fit(x_train ,y_train)
from sklearn.metrics import accuracy_score
print (accuracy_score(y_test, clf.predict(x_test)))
