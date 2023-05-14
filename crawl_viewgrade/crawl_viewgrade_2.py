from selenium import webdriver
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.remote.webelement import WebElement
from time import sleep


def crawl_grade():
    options = webdriver.ChromeOptions()
    options.add_experimental_option('prefs', {
        # Change default directory for downloads
        "download.default_directory": r"C:\Users\a\Documents\GitHub\ExtractTable\crawl_viewgrade\grade\2022-2023-1/",
        "download.prompt_for_download": False,  # To auto download the file
        "download.directory_upgrade": True,
        # It will not show PDF directly in chrome
        "plugins.always_open_pdf_externally": True
    })

    driver = webdriver.Chrome(
        executable_path="./chromedriver.exe", options=options)
    driver.get("http://112.137.129.30/viewgrade/#list")
    driver.find_element(By.ID, "username").send_keys("20020039")
    driver.find_element(By.ID, "password").send_keys("dublue1234")
    driver.find_element(By.ID, "loginButton").click()
    driver.find_element(By.XPATH, "//*[@id=\"button-choose-list\"]").click()
    sleep(2)
    select = Select(driver.find_element(
        By.ID, 'form-choose-year-list-subject'))
    select.select_by_visible_text("2022-2023")
    sleep(2)
    # form-choose-term-list-subject
    select = Select(driver.find_element(
        By.ID, 'form-choose-term-list-subject'))
    select.select_by_visible_text("Học kỳ 1")
    sleep(3)
    tbody = driver.find_element(By.ID, "needElementSubjectList")
    rows = tbody.find_elements(By.TAG_NAME, "tr")

    # print(len(rows))
    r = open(".\crawl_viewgrade\last.txt", "r")
    oldList = int(r.readline())
    update = 0
    if (len(rows) > oldList):
        update = len(rows) - oldList
    # print(update)
    w = open(".\crawl_viewgrade\last.txt", "w")
    w.write(str(len(rows)))
    cnt = 0
    for row in rows:
        if (cnt == update):
            break
        cnt = cnt+1
        a = row.find_element(By.TAG_NAME, "a")
        link = a.get_attribute("href")
        newGrade = link[46:-4]
        newGrade = newGrade.replace("%20", " ")

        if (cnt == 1):
            w = open(".\crawl_viewgrade\listNewGrade.txt", "w")
            w.write(newGrade)
            w.write("\n")
            w.write(link)
            w.write("\n")
        else:
            w = open(".\crawl_viewgrade\listNewGrade.txt", "a")
            w.write(newGrade)
            w.write("\n")
            w.write(link)
            w.write("\n")

        if a.get_attribute("class") == "not_link_to_mark_file":
            continue
        a.click()
        sleep(5)
    sleep(5)
