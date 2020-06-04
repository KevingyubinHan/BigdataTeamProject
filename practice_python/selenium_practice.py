# selenium_practice.py

from selenium import webdriver
from bs4 import BeautifulSoup

path = r'D:\Documents\MyDocuments\chromedriver.exe'
driver = webdriver.Chrome(path)

driver.get('https://search.konacard.co.kr/payable-merchants')
driver.find_element_by_xpath('//*[@id="kalist_area"]/div[1]/ul/li[11]').click()
# 우클릭-copy xpath
driver.find_element_by_class_name('btn_select.btn_black').click()

element = driver.find_element_by_name('searchKey')
element.clear()
element.send_keys('필라테스앤드')
driver.find_element_by_class_name('btn_search.btn_black_s').click()