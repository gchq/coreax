for filename in ./integration/test_*.py; do
    python -m pytest $filename
done
