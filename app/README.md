# README

Adaptive preference experiment.

**Note:** If you encounter any errors during the setup process, please refer to the [Troubleshooting Common Errors](#troubleshooting-common-errors) section at the end of this guide for solutions and workarounds.

### How do I get set up?

Follow these steps to set up your development environment. This guide assumes you are starting with a basic understanding of terminal commands and have the necessary permissions to install software on your machine.

1. **Create a Python 3.7 virtual environment in _backend/_**

   ```bash
   cd ${PROJECT}/RA/backend/
   python3.7 -m venv venv
   cd ${PROJECT}/RA/
   npm install
   ```

2. **Install requirements**

   ```bash
   cd ${PROJECT}/RA/backend/
   pip install -r requirements.txt
   ```

3. **Set the `ADAPTIVE_PREFERENCE_ENV` environment variable**

   For a development environment, add the following line to your `~/.bashrc` or `~/.zshrc` file:

   ```bash
   export ADAPTIVE_PREFERENCE_ENV="dev"
   ```

   Then, apply the changes:

   ```bash
   source ~/.bashrc
   # or
   source ~/.zshrc
   ```

## Running App

You will need to open two terminals or IDEs one for front end and one for back end.

- **Backend**

  ```bash
  cd ${PROJECT}/app/backend/
  source venv/bin/activate
  flask run
  ```

- **Frontend**
  For local development:

  ```bash
  cd ${PROJECT}/app/
  REACT_APP_BACKEND_ENDPOINT=http://127.0.0.1:5000 npm start
  ```

### Troubleshooting Common Errors

#### OpenSSL Error on Node.js

If you encounter an error similar to the following when starting the development server with `npm start`:

```
Error: error:0308010C:digital envelope routines::unsupported
    at new Hash (node:internal/crypto/hash:68:19)
    ...
Error: error:0308010C:digital envelope routines::unsupported
    ...
{ opensslErrorStack: [ 'error:03000086:digital envelope routines::initialization error' ],
  library: 'digital envelope routines',
  reason: 'unsupported',
  code: 'ERR_OSSL_EVP_UNSUPPORTED'
}
```

**Resolution:**

This error is due to Node.js version 17 and above using OpenSSL 3.0. To resolve this issue for development, set the `NODE_OPTIONS` environment variable to use OpenSSL's legacy provider:

```bash
export NODE_OPTIONS=--openssl-legacy-provider
```

Run the above command in your terminal before starting your development server. This workaround allows using cryptographic features unsupported by OpenSSL 3.0, enabling your server to start successfully.

#### Setup on M1/M2 Macs

M1/M2 Mac users may encounter issues installing Python 3.7 due to architecture compatibility. Here's a workaround using `conda`:

1. **Install Miniconda:** If you don’t have it, install Miniconda via brew for easier management.
2. **Open Terminal with Rosetta:** Navigate to your Terminal app, right-click > Get Info > Check "Open with Rosetta".
3. **Create and Configure Conda Environment:**

```bash
# Create a new conda environment named py37
conda create -n py37

# Activate the newly created environment
conda activate py37

# Configure conda to use x86_64 architecture
conda config --env --set subdir osx-64

# Install Python 3.7 in the environment
conda install python=3.7

# Install required Python packages
pip install -r requirements.txt
```

Source for workaround: [Stack Overflow](https://stackoverflow.com/questions/70205633/cannot-install-python-3-7-on-osx-arm64)

**Alternative using `pyenv`:**

If the above method does not work, consider using `pyenv` with the following command to install Python 3.7 under Rosetta 2:

```bash
arch -x86_64 pyenv install 3.7.17
```

Ensure you've installed `pyenv` and run this command in a Terminal opened with Rosetta to force installation using the Intel architecture.

---

### Who do I talk to?

[Ryan Dew](mailto:ryandew@wharton.upenn.edu)
