# Mark non-file targets as PHONY
.PHONY: all beautify

beautify:
	black -t py36 -l 120 *.py
	isort --atomic --multi-line 3 --trailing-comma --force-grid-wrap 0 --use-parentheses --line-width 120 --apply *.py
