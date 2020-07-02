# Welcome to your VS Code Extension

## What's in the folder

* This folder contains all of the files necessary for your color theme extension.
* `package.json` - this is the manifest file that defines the location of the theme file and specifies the base theme of the theme.
* `themes/cobalt9-color-theme.json` - the color theme definition file.

## Get up and running straight away

* Press `F5` to open a new window with your extension loaded.
* Open `File > Preferences > Color Themes` and pick your color theme.
* Open a file that has a language associated. The languages' configured grammar will tokenize the text and assign 'scopes' to the tokens. To examine these scopes, invoke the `Inspect TM Scopes` command from the Command Palette (`Ctrl+Shift+P` or `Cmd+Shift+P` on Mac) .

## Make changes

* Changes to the theme file are automatically applied to the Extension Development Host window.

## Adopt your theme to Visual Studio Code

* The token colorization is done based on standard TextMate themes. Colors are matched against one or more scopes.

To learn more about scopes and how they're used, check out the [color theme](https://code.visualstudio.com/api/extension-guides/color-theme) documentation.

## Install your extension

* To start using your extension with Visual Studio Code copy it into the `<user home>/.vscode/extensions` folder and restart Code.
* To share your extension with the world, read on https://code.visualstudio.com/docs about publishing an extension.


---
## Publishing Extensions

```sh
npm install -g vsce
```

<https://docs.microsoft.com/en-us/azure/devops/organizations/accounts/create-organization?view=azure-devops>

Go to <https://dev.azure.com/pydemia/>,
Then create Personal Access Token:
(Finally, scroll down the list of possible scopes until you find Marketplace and select Manage:)

```sh
cd cobalt9
# vsce create-publisher pydemia
vsce login pydemia
#Personal Access Token for publisher 'pydemia': ****************************************************
vsce package
# DONE  Packaged: /mnt/hdc1/data/git/pydemia-vscode-syntax/cobalt9/cobalt9-1.3.0.vsix (16 files, 469.37KB)
vsce publish
#Publishing pydemia.cobalt9@1.3.0...
# DONE  Published pydemia.cobalt9@1.3.0
#Your extension will live at https://marketplace.visualstudio.com/items?itemName=pydemia.cobalt9 (might take a few minutes for it to show up).
```