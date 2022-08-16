import time
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
import re
import requests
from bs4 import BeautifulSoup as soup
import pandas as pd
"""
Aim of this script is to get data for the analysis from the web site.

It is done in following way:

1. selenium opens website (if you want to make analysis for other team you need to change link in driver.get
Next it accept cookies and load ajax to have full list of meetings. Then it saves links to meetings

2. the links to meetings are used by beautifulsoup which opens every meeting and scrap it

3. the scrapped data are saved as following:

- whole data: 'raw_data.csv'
- matches data: 'raw_data_matches.csv'
- players data: 'raw_data_players.csv'

In order to run this script you need have chrome webdriver installed

"""


#poniższe otwiera stronę i akceptuje cookies. otwiera jeszcze więcej meczów oraz ściąga wszyskie numery meetingów w których brała udział Devka

PATH = r"C:\Users\rlszwa\Documents\chromedriver.exe"
driver = webdriver.Chrome(PATH)
try:
    driver.get("https://playarena.pl/46422,amatorskie,druzyny,sportowe,w_twoim_miescie,druzyna.html#meetings/")
    cookiesaccept = driver.find_element_by_link_text('Akceptuję')
    cookiesaccept.click()
    actions = ActionChains(driver)
    element = driver.find_element_by_id("ajax_team_46422_last_meetings")
    actions.move_to_element(element).perform()
    time.sleep(5)
    driver.find_element_by_id("ajax_team_46422_last_meetings").click()
    time.sleep(5)
    elem = driver.find_element_by_xpath("//*")
    source_code = elem.get_attribute("outerHTML")
except:
    pass
driver.quit()

"""get and save captain name"""

pattern = "kapitan(.*?). Załóż profil Twojej drużyny"
cpt_name = re.search(pattern, source_code).group(1)
f = open("cpt_name.txt", "a")
f.write(cpt_name)
f.close()

meetings = re.findall(r"meeting_line_(\d{6})", source_code)


mecze = []
players = []


for meeting in meetings:

    # %%

    response = requests.get('https://playarena.pl/meeting/' + meeting, verify=False)

    html_soup = soup(response.text, 'html.parser')
    # %%

    team_r = html_soup.find(class_='meetingInfoItem_right col-xs-6')
    team_r_name = team_r.find('a', attrs={'class': 'meetingInfoItemTeam'}).text
    team_r_rank = team_r.find('span', attrs={'class': 'font300'}).text

    team_r_score = team_r.find('span').text[1:]
    team_l = html_soup.find(class_='meetingInfoItem_left col-xs-6')
    team_l_score = team_l.find('span').text[1:]
    team_l_name = team_l.find('a', attrs={'class': 'meetingInfoItemTeam'}).text
    team_l_rank = team_l.find('span', attrs={'class': 'font300'}).text
    try:
        league = html_soup.find(class_='col-md-2 col-sm-12 text-center padding20').a.text
    except AttributeError:
        league = "no league"
    stadium = html_soup.find(class_='col-md-10 col-sm-12 padding20').a.text

    time_date = html_soup.find('dl', attrs={'class': 'dl-horizontal'}).dd.find_next('dd').text
    time = time_date[-6:]
    date = time_date[1:11]
    weekday = time_date[13:16]

    man_on_pithc = html_soup.find(class_='col-md-10 col-sm-12 padding20').find_next(class_='col-xs-4').find_next(
        class_='dl-horizontal').find_all('dd')[0].text
    match_lenght = html_soup.find(class_='col-md-10 col-sm-12 padding20').find_next(class_='col-xs-4').find_next(
        class_='dl-horizontal').find_all('dd')[1].text

    result_input = html_soup.find(string=re.compile("Wynik wprowadzony przez")).encode('utf-8')[26:]

    # %%

    squads = html_soup.find_all('a', attrs={'class': 'c_default'})

    names = []
    posess = []
    mvps = []
    gols = []
    teams = []
    pos_ads = []
    position_dic = {'Bramkarz': 1,
                    'Obrońca': 2,
                    'Pomocnik': 3,
                    'Napastnik': 4,
                    'Zawodnicy bez pozycji': 5}

    for player in squads[3:]:
        try:
            name = player.text
            names.append(name)
            pos = player.find_previous('h3').text
            posess.append(pos)
            mvp = player.find_next('td', attrs={'class': 'text-right'}).find('span')
            mvps.append(mvp)
            gol = player.find_next('td', attrs={'class': 'text-right'}).text
            gols.append(gol)
        except AttributeError:
            continue

    df = pd.DataFrame({'name': names,
                       'position': posess,
                       'mvp': mvps,
                       'scores': gols, })

    df['pos_no'] = df['position'].map(position_dic)

    # %%przyporządkowuje druzynę zawodnikowi
    df_dif = df['pos_no'].diff()

    for item, i in enumerate(df_dif):
        if i < 0:
            team_2 = item

    l = df['pos_no']

    teams = []

    for i, item in enumerate(l - 1):
        if i >= team_2:
            teams.append(team_l_name)
        else:
            teams.append(team_r_name)

    df['teams'] = teams
    df['time_date'] = time_date

    #position = df['position'].tolist()
    df = df[['time_date','name', 'position', 'mvp', 'scores', 'teams']]
    #list_fin = [date, league, man_on_pithc, match_lenght, stadium, team_l_name, team_l_rank, team_l_score,
    #            team_r_name, team_r_rank, team_r_score, time, weekday, result_input, names, position, mvps, gols, teams]
    list_fin = [time_date,date, league, man_on_pithc, match_lenght, stadium, team_l_name, team_l_rank, team_l_score,
                team_r_name, team_r_rank, team_r_score, time, weekday, result_input]
    mecze.append(list_fin)
    players.append(df)



mecze_df = pd.DataFrame(mecze)
col_names = ['time_date', 'date', 'league', 'man_on_pitch', 'match_lenght', 'stadium', 'team_l_name', 'team_l_rank', 'team_l_score',
                'team_r_name', 'team_r_rank', 'team_r_score', 'time', 'weekday', 'result_input']
mecze_df.columns = col_names

mecze_df.to_csv('raw_data_matches.csv')

players_df = pd.concat(players)
players_df.to_csv('raw_data_players.csv')
players_df = pd.DataFrame(players_df)
players_df.to_excel('raw_data_players.xlsx')

raw_data = pd.merge(players_df, mecze_df, on = 'time_date', how = 'left', sort= False)
raw_data.to_csv('raw_data.csv')
raw_data.to_excel('raw_data.xlsx')
