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

import org.edgexfoundry.controller.integration.AddressableControllerTest;
import org.edgexfoundry.controller.integration.CallbackExecutorTest;
import org.edgexfoundry.controller.integration.CommandControllerTest;
import org.edgexfoundry.controller.integration.DeviceControllerTest;
import org.edgexfoundry.controller.integration.DeviceProfileControllerTest;
import org.edgexfoundry.controller.integration.DeviceReportControllerTest;
import org.edgexfoundry.controller.integration.DeviceServiceControllerTest;
import org.edgexfoundry.controller.integration.ProvisionWatcherControllerTest;
import org.edgexfoundry.controller.integration.ScheduleControllerTest;
import org.edgexfoundry.controller.integration.ScheduleEventControllerTest;
import org.edgexfoundry.dao.integration.AddressableDaoTest;
import org.edgexfoundry.dao.integration.AddressableRepositoryTest;
import org.edgexfoundry.dao.integration.CommandRepositoryTest;
import org.edgexfoundry.dao.integration.DeviceDaoTest;
import org.edgexfoundry.dao.integration.DeviceProfileDaoTest;
import org.edgexfoundry.dao.integration.DeviceReportDaoTest;
import org.edgexfoundry.dao.integration.DeviceReportRepositoryTest;
import org.edgexfoundry.dao.integration.DeviceRepositoryTest;
import org.edgexfoundry.dao.integration.DeviceServiceDaoTest;
import org.edgexfoundry.dao.integration.DeviceServiceRepositoryTest;
import org.edgexfoundry.dao.integration.ProvisionWatcherRepositoryTest;
import org.edgexfoundry.dao.integration.ScheduleDaoTest;
import org.edgexfoundry.dao.integration.ScheduleEventDaoTest;
import org.edgexfoundry.dao.integration.ScheduleEventRepositoryTest;
import org.edgexfoundry.dao.integration.ScheduleRepositoryTest;
import org.edgexfoundry.integration.mongodb.MongoDBConnectivityTest;
import org.edgexfoundry.integration.spring.SpringConfigurationTest;
import org.junit.Ignore;
import org.junit.runner.RunWith;
import org.junit.runners.Suite;

/**
 * Used in development only. Remove @Ignore to run just the integration tests (not unit tests).
 * These tests do require other resources to run.
 * 
 * @author Jim White
 *
 */
@Ignore
@RunWith(Suite.class)
@Suite.SuiteClasses({AddressableControllerTest.class, CallbackExecutorTest.class,
    CommandControllerTest.class, DeviceControllerTest.class, DeviceProfileControllerTest.class,
    DeviceReportControllerTest.class, DeviceServiceControllerTest.class,
    ProvisionWatcherControllerTest.class, ScheduleControllerTest.class,
    ScheduleEventControllerTest.class, AddressableDaoTest.class, AddressableRepositoryTest.class,
    CommandRepositoryTest.class, DeviceDaoTest.class, DeviceProfileDaoTest.class,
    DeviceReportDaoTest.class, DeviceReportRepositoryTest.class, DeviceRepositoryTest.class,
    DeviceServiceDaoTest.class, DeviceServiceRepositoryTest.class,
    ProvisionWatcherRepositoryTest.class, ScheduleDaoTest.class, ScheduleEventDaoTest.class,
    ScheduleEventRepositoryTest.class, ScheduleEventRepositoryTest.class,
    ScheduleRepositoryTest.class, MongoDBConnectivityTest.class, SpringConfigurationTest.class})
public class IntegrationTestSuite {

}
