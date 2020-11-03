from sqlalchemy import create_engine
from selenium import webdriver
import time
from bs4 import BeautifulSoup
import pandas as pd
import re

def clean_review(review):
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", review).split())


def page_filter(page_src, element_type, class_type):
    soup = BeautifulSoup(page_src, 'html.parser')
    reviews = soup.find_all(element_type, class_=class_type)
    return(reviews)


def get_the_reviews(page_src,review_sort,label,reviews):   
    for elements in page_src:
        if review_sort in str(elements):
            filtered_page_src = page_filter(page_src=str(elements), element_type='span', class_type='c-review__body')
            for review_element in filtered_page_src:
                print(clean_review(review_element.get_text()))
                reviews["reviews"].append(clean_review(review_element.get_text()))
                reviews["label"].append(label)


def page_number(driver):
    # find the page number
    pageNumSrc = page_filter(page_src=driver.page_source, element_type='div', class_type='bui-pagination__list')
    return [int(pageNum.get_text().split('Page')[-1].replace('\n','')) for pageNum in pageNumSrc][0]


def web_scraping(filter, review_sort,label,reviews):
    #begin with the second page because the first page is already has been opened
    counter= 1
    while True:
        #Click on the filter to get only the poor reviews
        if counter == 1: 
            driver.find_element_by_xpath("//select[contains(@id, 'review_score_filter')]/option[contains(text(), '"+filter+"')]").click()
            time.sleep(1)
            pageNumber=page_number(driver=driver)
        time.sleep(0.5)
        page = page_filter(page_src=driver.page_source, element_type='p', class_type='c-review__inner c-review__inner--ltr')
        get_the_reviews(page_src=page, review_sort=review_sort, label=label, reviews=reviews)
        time.sleep(0.5)
        if counter == pageNumber:
            counter = 1
            print("break")
            break
        print(counter, pageNumber)
        # time.sleep(0.5)
        driver.find_element_by_xpath("//a[contains(@class, 'pagenext')]").click()
        counter +=1

def to_database(reviews): 
    df = pd.DataFrame.from_dict(reviews)
    engine = create_engine('mysql+mysqlconnector://root:*****@127.0.0.1:3306/test')
    df.to_sql(name='reviews_scraped', con=engine,
            if_exists='replace', index=False, method='multi')
    df.to_csv (r'.\reviews_scraped.csv', index = False, header=True)


if __name__ == "__main__":
    websites= [
    "https://www.booking.com/hotel/nl/victoria.html?label=gen173nr-1DCAEoggI46AdIM1gEaKkBiAEBmAExuAEXyAEM2AED6AEB-AECiAIBqAIDuALz5Nf8BcACAdICJDc3NmY5Njg2LWMxMTItNDRiZC05NzM2LTQ4OGVkMTRhYjQ0ZNgCBOACAQ;sid=5865ce8ee522ea36adf0fccfd6e0b3ac;dest_id=-2140479;dest_type=city;dist=0;group_adults=2;group_children=0;hapos=3;hpos=3;no_rooms=1;room1=A%2CA;sb_price_type=total;sr_order=popularity;srepoch=1603662466;srpvid=2a3d9940f4ca0192;type=total;ucfs=1&#tab-reviews",
    "https://www.booking.com/hotel/nl/nadia-amsterdam.html?label=gen173nr-1DCAEoggI46AdIM1gEaKkBiAEBmAExuAEXyAEM2AED6AEB-AECiAIBqAIDuALz5Nf8BcACAdICJDc3NmY5Njg2LWMxMTItNDRiZC05NzM2LTQ4OGVkMTRhYjQ0ZNgCBOACAQ;sid=5865ce8ee522ea36adf0fccfd6e0b3ac;dest_id=145;dest_type=district;dist=0;group_adults=2;group_children=0;hapos=14;hpos=14;no_rooms=1;room1=A%2CA;sb_price_type=total;sr_order=popularity;srepoch=1603689814;srpvid=762525eb98a40033;type=total;ucfs=1&#tab-reviews",
    "https://www.booking.com/hotel/nl/oldnickeladam.html?aid=304142;label=gen173nr-1DCAEoggI46AdIM1gEaKkBiAEBmAExuAEXyAEM2AED6AEB-AECiAIBqAIDuALz5Nf8BcACAdICJDc3NmY5Njg2LWMxMTItNDRiZC05NzM2LTQ4OGVkMTRhYjQ0ZNgCBOACAQ;sid=5865ce8ee522ea36adf0fccfd6e0b3ac;dest_id=145;dest_type=district;dist=0;group_adults=2;group_children=0;hapos=29;hpos=4;no_rooms=1;room1=A%2CA;sb_price_type=total;sr_order=popularity;srepoch=1603689886;srpvid=20cf260fb06800a8;type=total;ucfs=1&#tab-reviews"
]
    reviews = {"reviews": [], "label": []}
    for website in websites:
        driver = webdriver.Chrome("C:/inviduBigData/chromedriver")
        driver.get(website)
        time.sleep(1)

        # Refuse the cookies to don't have error while clicking.
        driver.find_element_by_xpath("//button[contains(@id, 'onetrust-reject-all-handler') and text() = 'Decline']").click()

        #Click on the English button to show only english reviews
        driver.find_element_by_xpath("//div[contains(@class, 'bui-group bui-group--inline language_filter_value_row')]/label[contains(@class,'bui-input-checkbutton')][2]").click()
        time.sleep(1.5)

        web_scraping(filter="Very Poor", review_sort="review_poor", label=0, reviews=reviews)
        web_scraping(filter="Poor", review_sort="review_poor", label=0, reviews=reviews)
        web_scraping(filter="Wonderful", review_sort="review_great", label=1, reviews=reviews)

    # to_database(reviews=reviews)



