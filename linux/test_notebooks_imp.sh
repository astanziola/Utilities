source "$HOME/anaconda3/etc/profile.d/conda.sh"

# Test privacy notebooks
echo "Test privacy notebooks (skipping Competition 2)"
conda activate privacy
cd ~/privacy-exercise

# Throw an error if fails, time the execution of the command
start_time=$(date +%s)
jupyter nbconvert --execute _test_Exercise.ipynb --to python 2> /dev/null || exit 1
end_time=`date +%s`
echo "Correctly executed notebook _test_Exercise.ipynb in $((end_time-start_time)) seconds"

start_time=$(date +%s)
jupyter nbconvert --execute _test_Competition_1.ipynb --to python 2> /dev/null || exit 1
end_time=`date +%s`
echo "Correctly executed notebook _test_Competition_1.ipynb in $((end_time-start_time)) seconds"
echo "-- All notebooks of the privacy exercise have been executed successfully"

# Remove generated python scripts
rm _test_Competition_1.py
rm _test_Exercise.py

# Test cbir
echo "Test Scalable Recognition exercise"
conda activate cbir
cd ~/scalable-recognition-with-a-vocabulary-tree

# Run notebooks
echo "Downloading data before measuring execution time"
python cbir/download.py 2> /dev/null || exit 1
start_time=$(date +%s)
jupyter nbconvert --execute '_test_Scalable Recognition with a Vocabulary Tree copy.ipynb' --to python 2> /dev/null || exit 1
end_time=`date +%s`
echo "Correctly executed notebook _test_Scalable Recognition with a Vocabulary Tree copy.ipynb in $((end_time-start_time)) seconds"
echo "-- All notebooks of the Scalable Recognition exercise have been executed successfully"

# remove generated python scripts
rm '_test_Scalable Recognition with a Vocabulary Tree copy.py'

# All tests passed
echo "-- All tests passed"