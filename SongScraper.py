"""

This little piece of code iterates through different URL's, each one containing
one song by Hip Hop artist Notorious BIG and creates a .txt file to store all
the lyrics and titles

"""


import requests
import re
from bs4 import BeautifulSoup

with open('nBIGsongs.txt', 'a') as f:

    r = requests.get("http://www.metrolyrics.com/notorious-big-albums-list.html")
    c = r.content
    soup = BeautifulSoup(c, "html.parser")

    albums = soup.find("div", {'class' : 'grid_8'})


    for a in albums.find_all('a', href=True, alt=True):
        r = requests.get(a['href'])
        c = r.content
        soup = BeautifulSoup(c, "html.parser")
        song = soup.find_all('p', {'class':'verse'})
        title = soup.find_all('h1')

        for item in title:
            title = item.text.replace('Lyrics','')
            f.write("\n" + title.upper() + "\n")

        for item in song:
            f.write(item.text)

f.close()
