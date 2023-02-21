Commands to setup project

- Install Expo CLI: 
    - `npm install -g expo-cli`
- Install Yarn
    - `npm install -g yarn`
- Install Project dependencies
    - `yarn`
    - `yarn install`
- Create ios and android directories project files
    - `expo prebuild`
- Retrieve the pods for the project
    - `cd ios`
    - `pod install`
    - `cd ..` 
- Perform a native build
    - iOS: `expo run:ios` 
    - android: `expo run:android`