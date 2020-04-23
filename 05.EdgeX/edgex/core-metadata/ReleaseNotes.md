# v0.2 (10/20/2017)
# Release Notes

## Notable Changes
The Barcelona Release (v 0.2) of the Core Metadata micro service includes the following:
* Application of Google Style Guidelines to the code base
* Increase in unit/intergration tests from 630 tests to 965 tests
* POM changes for appropriate repository information for distribution/repos management, checkstyle plugins, etc.
* Removed all references to unfinished DeviceManager work as part of Dell Fuse
* Added Dockerfile for creation of micro service targeted for ARM64 
* Added interfaces for all Controller classes

## Bug Fixes
* Fix difference in API versus RAML document
* GET Device Profile by id not returning 404
* Removed OS specific file path for logging file 
* Provide option to include stack trace in log outputs

## Pull Request/Commit Details
 - [#15](https://github.com/edgexfoundry/core-metadata/pull/15) - Remove staging plugin contributed by Jeremy Phelps ([JPWKU](https://github.com/JPWKU))
 - [#14](https://github.com/edgexfoundry/core-metadata/pull/14) - Fixes Maven artifact dependency path contributed by Tyler Cox ([trcox](https://github.com/trcox))
 - [#13](https://github.com/edgexfoundry/core-metadata/pull/13) - added staging and snapshots repos to pom along with nexus staging mav… contributed by Jim White ([jpwhitemn](https://github.com/jpwhitemn))
 - [#12](https://github.com/edgexfoundry/core-metadata/pull/12) - Add aarch64 docker file contributed by ([feclare](https://github.com/feclare))
 - [#11](https://github.com/edgexfoundry/core-metadata/pull/11) - Adds Docker build capability contributed by Tyler Cox ([trcox](https://github.com/trcox))
 - [#10](https://github.com/edgexfoundry/core-metadata/pull/10) - removed duplicate property for nexus in pom.xml contributed by Jim White ([jpwhitemn](https://github.com/jpwhitemn))
 - [#9](https://github.com/edgexfoundry/core-metadata/pull/9) - added tests, pom updates for checkstyles and nexus, addition of controller interfaces contributed by Jim White ([jpwhitemn](https://github.com/jpwhitemn))
 - [#8](https://github.com/edgexfoundry/core-metadata/pull/8) - checkstyle changes, lint changes, package restructure, addition of in… contributed by Jim White ([jpwhitemn](https://github.com/jpwhitemn))
 - [#7](https://github.com/edgexfoundry/core-metadata/issues/7) - Error responses to REST API calls include full stack traces
 - [#6](https://github.com/edgexfoundry/core-metadata/issues/6) - RAML API issue for device +fix
 - [#5](https://github.com/edgexfoundry/core-metadata/issues/5) - GET Device Profile by id not returning 404 +fix
 - [#4](https://github.com/edgexfoundry/core-metadata/pull/4) - Fixes Log File Path contributed by Tyler Cox ([trcox](https://github.com/trcox))
 - [#3](https://github.com/edgexfoundry/core-metadata/issues/3) - Log File Path not Platform agnostic
 - [#2](https://github.com/edgexfoundry/core-metadata/pull/2) - Add distributionManagement for artifact storage contributed by Andrew Grimberg ([tykeal](https://github.com/tykeal))
 - [#1](https://github.com/edgexfoundry/core-metadata/pull/1) - Contributed Project Fuse source code contributed by Tyler Cox ([trcox](https://github.com/trcox))

