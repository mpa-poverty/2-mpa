# Set up a new EE account, step-by-step guide

### Sign-up for Earth-Engine
Before going further, you need to register an Google Earth Engine account @ https://code.earthengine.google.com using any gmail address you have access to. This process may take a couple of days. Without registration, the `ee.Initialize()` will not work. And this is mandatory to use the earth-engine API to access, handle and download the data.

### Link Your Personal Drive
The earth-engine API will only allow exporting the data to your service account's drive, which, by default, is not linked to your personal drive. 
To do so, and properly have access to the exported data, we need to enable the drive API in within the service accounts' settings.
