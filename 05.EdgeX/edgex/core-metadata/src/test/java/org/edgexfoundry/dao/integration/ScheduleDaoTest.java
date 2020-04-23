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
 * @microservice: core-metadata
 * @author: Jim White, Dell
 * @version: 1.0.0
 *******************************************************************************/

package org.edgexfoundry.dao.integration;

import static org.edgexfoundry.test.data.ScheduleData.TEST_SCHEDULE_NAME;
import static org.edgexfoundry.test.data.ScheduleData.TEST_SCHEDULE_NAME_NONE;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import java.util.List;

import org.edgexfoundry.Application;
import org.edgexfoundry.dao.AddressableRepository;
import org.edgexfoundry.dao.DeviceReportRepository;
import org.edgexfoundry.dao.DeviceRepository;
import org.edgexfoundry.dao.DeviceServiceRepository;
import org.edgexfoundry.dao.ScheduleDao;
import org.edgexfoundry.dao.ScheduleEventRepository;
import org.edgexfoundry.dao.ScheduleRepository;
import org.edgexfoundry.domain.meta.Addressable;
import org.edgexfoundry.domain.meta.Asset;
import org.edgexfoundry.domain.meta.Device;
import org.edgexfoundry.domain.meta.DeviceReport;
import org.edgexfoundry.domain.meta.DeviceService;
import org.edgexfoundry.domain.meta.Schedule;
import org.edgexfoundry.domain.meta.ScheduleEvent;
import org.edgexfoundry.test.category.RequiresMongoDB;
import org.edgexfoundry.test.category.RequiresSpring;
import org.edgexfoundry.test.category.RequiresWeb;
import org.edgexfoundry.test.data.AddressableData;
import org.edgexfoundry.test.data.DeviceData;
import org.edgexfoundry.test.data.ReportData;
import org.edgexfoundry.test.data.ScheduleData;
import org.edgexfoundry.test.data.ScheduleEventData;
import org.edgexfoundry.test.data.ServiceData;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.experimental.categories.Category;
import org.junit.runner.RunWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.SpringApplicationConfiguration;
import org.springframework.test.context.junit4.SpringJUnit4ClassRunner;
import org.springframework.test.context.web.WebAppConfiguration;

@RunWith(SpringJUnit4ClassRunner.class)
@SpringApplicationConfiguration(classes = Application.class)
@WebAppConfiguration("src/test/resources")
@Category({RequiresMongoDB.class, RequiresSpring.class, RequiresWeb.class})
public class ScheduleDaoTest {

  @Autowired
  private ScheduleDao dao;

  @Autowired
  private ScheduleRepository repos;

  @Autowired
  private ScheduleEventRepository schEventRepos;

  @Autowired
  private AddressableRepository addrRepos;

  @Autowired
  private DeviceServiceRepository serviceRepos;

  @Autowired
  private DeviceReportRepository rptRepos;

  @Autowired
  private DeviceRepository deviceRepos;

  private String id;

  @Before
  public void createTestData() {
    DeviceService srv = ServiceData.newTestInstance();
    serviceRepos.save(srv);
    Device device = DeviceData.newTestInstance();
    device.setService(srv);
    deviceRepos.save(device);
    Addressable addr = AddressableData.newTestInstance();
    addrRepos.save(addr);
    ScheduleEvent event = ScheduleEventData.newTestInstance();
    event.setAddressable(addr);
    schEventRepos.save(event);
    DeviceReport report = ReportData.newTestInstance();
    rptRepos.save(report);
    Schedule schedule = ScheduleData.newTestInstance();
    repos.save(schedule);
    id = schedule.getId();
    assertNotNull("new test Schedule has no identifier", id);
    assertNotNull("Dao is null", dao);
  }

  @After
  public void cleanup() {
    deviceRepos.deleteAll();
    rptRepos.deleteAll();
    addrRepos.deleteAll();
    schEventRepos.deleteAll();
    serviceRepos.deleteAll();
    repos.deleteAll();
  }

  @Test
  public void testGetByIdOrName() {
    Schedule schedule = ScheduleData.newTestInstance();
    schedule.setName(TEST_SCHEDULE_NAME);
    assertNotNull("Schedule is null on getByIdOrName with valid name", dao.getByIdOrName(schedule));
    schedule.setName(null);
    schedule.setId(id);
    assertNotNull("Schedule is null on getByIdOrName with valid id", dao.getByIdOrName(schedule));
  }

  @Test
  public void testGetByIdOrNameWithNull() {
    assertNull("No schedule should be found with null on getByIdOrName", dao.getByIdOrName(null));
  }

  @Test
  public void testGetByIdOrNameWithBadIdentifiers() {
    Schedule schedule = ScheduleData.newTestInstance();
    schedule.setId("badid");
    assertNull("No schedule should be found with bad id on getByIdOrName", dao.getByIdOrName(schedule));
    schedule.setId(null);
    schedule.setName("badname");
    assertNull("No schedule should be found with bad name on getByIdOrName", dao.getByIdOrName(schedule));
  }

  @Test
  public void testIsScheduleAssociatedToScheduleEvent() {
    assertTrue("Schedule should be associated to a ScheduleEvent",
        dao.isScheduleAssociatedToScheduleEvent(repos.findOne(id)));
  }

  @Test
  public void testIsScheduleAssociatedToScheduleEventWithNone() {
    schEventRepos.deleteAll();
    assertFalse("Schedule should not be associated to a ScheduleEvent",
        dao.isScheduleAssociatedToScheduleEvent(repos.findOne(id)));
  }

  @Test
  public void testIsScheduleAssociatedToScheduleEventWithNull() {
    assertFalse("Schedule should not be associated to a ScheduleEvent",
        dao.isScheduleAssociatedToScheduleEvent(null));
  }

  @Test
  public void testGetAffectedServices() {
    List<Asset> services = dao.getAffectedServices(repos.findOne(id));
    assertEquals("Service associated to schedule was not found", 1, services.size());
    assertEquals("Affected servcies does not include appropriate service",
        ServiceData.TEST_SERVICE_NAME, services.get(0).getName());
  }

  @Test
  public void testGetAffectedServicesWithNone() {
    Schedule schedule = repos.findOne(id);
    schedule.setName(TEST_SCHEDULE_NAME_NONE);
    schedule.setId(null);
    assertTrue("Affected services should be empty - none assocaited",
        dao.getAffectedServices(schedule).isEmpty());
  }

  @Test
  public void testGetAffectedServicesGivenNull() {
    assertTrue("Affected services should be empty - none assocaited",
        dao.getAffectedServices(null).isEmpty());
  }
}
