Steps to run the script:

### Team
- Aina Llaneras
- Maria Angelica Jaimes
- Xavier Querol 
- Bernat Comas


### Running script for the first time

1. Open folder in terminal
```bash
cd <root_folder_of_project>/
```
2. Create virtual env
```bash
python3 -m venv venv/
```
3. Open virtual env
```bash
source venv/bin/activate
```
4. Install required dependencies
```bash
pip install -r requirements.txt
```
you can check if dependencies were installed by running next
command,it should print list with installed dependencies
```bash
pip list
```

Execute main.py by running:
```bash
 python3 main.py
 ```

Then a menu will be displayed with the instructions to execute any combination of hyperparameters. Follow the instructions
- If you put in a wrong parameter the app will make you input it again.
- You can put both the number in the left side of the parameter and the name of the parameter itself as shown in the options.

Any model presented in this report can be run through the main menu.