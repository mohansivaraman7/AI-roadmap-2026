pipeline {
    agent any

    environment {
        VENV_DIR = 'venv'
    }

    options {
        timestamps()
        disableConcurrentBuilds()
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Set Up Python Environment') {
            steps {
                sh '''
                    python3 -m venv ${VENV_DIR}
                    . ${VENV_DIR}/bin/activate
                    pip install --upgrade pip
                    pip install -r requirements.txt
                    pip install requests
                '''
            }
        }

        stage('Train Model') {
            steps {
                sh '''
                    . ${VENV_DIR}/bin/activate
                    python train_model.py
                '''
            }
        }

        stage('Start API & Smoke Test') {
            steps {
                sh '''
                    . ${VENV_DIR}/bin/activate

                    nohup python app.py > app.log 2>&1 &
                    echo $! > app.pid

                    echo "Waiting for API to become ready..."
                    ready=0
                    for i in $(seq 1 30); do
                        if curl -s -o /dev/null http://127.0.0.1:5000/; then
                            ready=1
                            echo "API is up"
                            break
                        fi
                        sleep 1
                    done

                    if [ "$ready" -ne 1 ]; then
                        echo "API did not start in time"
                        cat app.log || true
                        exit 1
                    fi

                    python test_prediction.py
                '''
            }
        }
    }

    post {
        always {
            sh '''
                if [ -f app.pid ]; then
                    kill "$(cat app.pid)" 2>/dev/null || true
                    rm -f app.pid
                fi
                # Flask's debug-mode reloader forks a child process; make sure
                # nothing is left listening on the API port.
                fuser -k 5000/tcp 2>/dev/null || true
            '''
            archiveArtifacts artifacts: 'house_model.pkl, app.log', allowEmptyArchive: true
        }
        success {
            echo 'Build, train, and smoke test succeeded.'
        }
        failure {
            echo 'Pipeline failed — check app.log and the console output above for details.'
        }
        cleanup {
            sh 'rm -rf ${VENV_DIR}'
        }
    }
}
