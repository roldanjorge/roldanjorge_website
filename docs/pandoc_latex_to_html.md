
```bash
pandoc main.tex \
-f latex \
-t html \
 --mathjax --citeproc --metadata title="My post" -o index.html
```

```bash
pandoc input.tex \
-f latex \
-t html \
--mathjax \
--citeproc \
--metadata title="Title" \
-o index.html
```