from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import NoSuchElementException

import time
import pandas as pd
import numpy as np

def wait_long():
    return np.random.normal(10,0.3)

def wait_medium():
    return np.random.normal(4,0.3)

def get_list_links(driver):
    all_links_on_page_s = driver.find_elements_by_xpath('//*[@class="_15_ydu6b"]')
    all_links_on_page = [restau.get_attribute("href") for restau in all_links_on_page_s]
    return all_links_on_page

def get_number_comment_pages(driver):
    try:
        numb_pages_comments = driver.find_element_by_xpath('//*[contains(@class, "pageNum last")]').text
        return int(numb_pages_comments)
    except NoSuchElementException:
        return 0

def get_number_restaurant_pages(driver):
    numb_pages_restaurants = driver.find_element_by_xpath('//*[@class="pageNumbers"]').text.split("\n")[-1]
    return int(numb_pages_restaurants)

def identify_next_page_button_restaurants(driver):
    return driver.find_element_by_xpath('//*[@class="nav next rndBtn ui_button primary taLnk"]').get_attribute('href')

def identify_next_page_button_comments(driver):
    try:
        return driver.find_element_by_xpath('//*[@class="nav next ui_button primary  cx_brand_refresh_phase2"]')
    except NoSuchElementException:
        return None
    
def retrieve_comments(driver,numb_comments):
    try:
        lst_comment_boxes_s = driver.find_elements_by_xpath('//*[@class="ui_column is-9"]')
        lst_comments = []
        for i in lst_comment_boxes_s[:numb_comments]:
            lst_comments.append(i.find_elements_by_tag_name("div")[1].text.replace(',','').replace(';',''))
        return lst_comments
    except NoSuchElementException:
        return None

def retrieve_ratings(driver):
    try:
        lst_ratings_s = driver.find_elements_by_xpath('//*[@class="review-container"]//*[contains(@class,"ui_bubble_rating bubble_")]')
        lst_ratings = [int(i.get_attribute("class")[-2:])/10 for i in lst_ratings_s]
        return lst_ratings
    except NoSuchElementException:
        return None

def retrieve_bonus_info(driver):
    try:
        lst_elements_s = driver.find_elements_by_xpath('//*[@class="restaurants-detail-overview-cards-DetailOverviewCards__wrapperDiv--1Dfhf"]')
        return (' ').join(lst_elements_s[1].text.replace(',','').replace(';','').split("\n"))
    except NoSuchElementException:
        return None


from selenium.webdriver.chrome.options import Options
options = webdriver.ChromeOptions() 
options.add_argument("start-maximized")
driver = webdriver.Chrome(options=options)
driver.get("https://www.tripadvisor.fr/RestaurantSearch-g187147-oa180-a_date.2020__2D__05__2D__27-a_people.2-a_time.20%3A00%3A00-a_zur.2020__5F__05__5F__27-p6-Pari.html#EATERY_LIST_CONTENTS")

time.sleep(wait_long())

# button_repas_eco = driver.find_element_by_xpath('//*[text()="Repas économique"]')
# button_repas_int = driver.find_element_by_xpath('//*[text()="Intermédiaire"]')
# button_repas_cuisraff = driver.find_element_by_xpath('//*[text()="Cuisine raffinée"]')

# button_repas_eco.click()
# time.sleep(generate_random_time())
# button_repas_int.click()
# time.sleep(wait_long())
# button_repas_cuisraff.click()
# time.sleep(wait_long())

numb_pages_restaurants = get_number_restaurant_pages(driver)

masterdf = pd.DataFrame(columns=["comment","rating","bonus_info","city"])

for page in range(40):
    time.sleep(wait_long())
    links = get_list_links(driver)
    next_page_url = identify_next_page_button_restaurants(driver)
    print(next_page_url)
    for link in links:
        driver.get(link)
        time.sleep(wait_long())
        bonus_info = retrieve_bonus_info(driver)
        numb_comment_pages = get_number_comment_pages(driver)
        for comment_page in range(0,numb_comment_pages-1):
            try:
                lst_ratings = retrieve_ratings(driver)
                lst_comments = retrieve_comments(driver,len(lst_ratings))
                print(len(lst_comments),lst_comments)
                print(len(lst_ratings),lst_ratings)
                tempdf = pd.DataFrame({"comment": [comment for comment in lst_comments],
                                      "rating": [rating for rating in lst_ratings],
                                      })
                tempdf["bonus_info"]=str(bonus_info)
                tempdf["city"] = "Paris"
                masterdf= masterdf.append(tempdf)
                masterdf.to_csv("masterdf4_scrape.csv")
            except ValueError:
                pass
            next_button = identify_next_page_button_comments(driver)
            if next_button:
                next_button.click()
                time.sleep(wait_long())
            else:
                break
    time.sleep(wait_long())        
    driver.get(next_page_url)

driver.close()
