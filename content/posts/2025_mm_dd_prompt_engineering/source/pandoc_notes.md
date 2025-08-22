
```bash
pandoc the_math_of_lms.tex \
-f latex \
-t html \
--mathjax \
--citeproc \
--metadata title="The Math of Language Models" \
-o index.html
```