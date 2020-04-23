/*******************************************************************************
 * Copyright 2016-2017 Dell Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 *
 * @microservice: support-notifications
 * @author: Jim White, Dell
 * @version: 1.0.0
 *******************************************************************************/

package org.edgexfoundry.core.metadata.suites;

import org.edgexfoundry.controller.AddressableControllerTest;
import org.edgexfoundry.controller.CommandControllerTest;
import org.edgexfoundry.controller.DeviceControllerTest;
import org.edgexfoundry.controller.DeviceProfileControllerTest;
import org.edgexfoundry.controller.DeviceReportControllerTest;
import org.edgexfoundry.controller.DeviceServiceControllerTest;
import org.edgexfoundry.controller.PingControllerTest;
import org.edgexfoundry.controller.ProvisionWatcherControllerTest;
import org.edgexfoundry.controller.ScheduleControllerTest;
import org.edgexfoundry.controller.ScheduleEventControllerTest;
import org.edgexfoundry.dao.AddressableDaoTest;
import org.edgexfoundry.dao.DeviceDaoTest;
import org.edgexfoundry.dao.DeviceProfileDaoTest;
import org.edgexfoundry.dao.DeviceReportDaoTest;
import org.edgexfoundry.dao.DeviceServiceDaoTest;
import org.edgexfoundry.dao.integration.ScheduleDaoTest;
import org.edgexfoundry.dao.integration.ScheduleEventDaoTest;
import org.junit.Ignore;
import org.junit.runner.RunWith;
import org.junit.runners.Suite;

/**
 * Used in development only. Remove @Ignore to run just the unit tests (not integration tests).
 * These tests do require other resources to run.
 * 
 * @author Jim White
 *
 */
@Ignore
@RunWith(Suite.class)
@Suite.SuiteClasses({AddressableControllerTest.class, CommandControllerTest.class,
    DeviceControllerTest.class, DeviceProfileControllerTest.class, DeviceReportControllerTest.class,
    DeviceServiceControllerTest.class, PingControllerTest.class,
    ProvisionWatcherControllerTest.class, ScheduleControllerTest.class,
    ScheduleEventControllerTest.class, AddressableDaoTest.class, DeviceDaoTest.class,
    DeviceProfileDaoTest.class, DeviceReportDaoTest.class, DeviceServiceDaoTest.class,
    ScheduleDaoTest.class, ScheduleEventDaoTest.class})
public class UnitTestSuite {

}
