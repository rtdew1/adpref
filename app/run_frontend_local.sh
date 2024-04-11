if [ "$ADAPTIVE_PREFERENCE_ENV" == "dev-docker" ]
then
    REACT_APP_BACKEND_ENDPOINT='http://localhost:5555' npm start
elif [ "$ADAPTIVE_PREFERENCE_ENV" == "dev" ]
then
    REACT_APP_BACKEND_ENDPOINT='http://localhost:5000' npm start
else
    echo "Environment variable ADAPTIVE_PREFERENCE_ENV must be set to either 'dev' or 'dev-docker' for this script"
fi
