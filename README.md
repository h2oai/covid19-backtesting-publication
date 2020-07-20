# Q-Peak

This app allows to forecast confirmed cases, fatalities, and recoveries for Covid-19 in individual countries.
Please refer to our [documentation](q_peak/documentation/Documentation.md) for further details.

# Development

**Step 1.** Download and extract [Q](https://github.com/h2oai/q/releases).

**Step 2.** Clone this repo inside the apps folder.

```bash
cd q/apps
git clone https://github.com/h2oai/q-peak.git
```

**Step 3.** Set up the development environment.  

```bash
cd q-peak
make setup
```

**Step 4.** Run Q.

```bash
cd ../..
./qhttpd -conf q.toml # Linux
OR
./qhttpd-darwin -conf q.toml # OSX
```

# Notes
* The latest release is test with version 0.5

