
```bash
pandoc main.tex \
-f latex \
-t html \
 --mathjax --citeproc --metadata title="My post" -o index.html
```