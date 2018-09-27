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
import static org.edgexfoundry.test.data.ScheduleEventData.TEST_SCHEDULE_EVENT_NAME;
import static org.edgexfoundry.test.data.ScheduleEventData.checkTestData;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.util.List;

import org.edgexfoundry.Application;
import org.edgexfoundry.dao.AddressableRepository;
import org.edgexfoundry.dao.DeviceReportRepository;
import org.edgexfoundry.dao.DeviceRepository;
import org.edgexfoundry.dao.DeviceServiceRepository;
import org.edgexfoundry.dao.ScheduleEventRepository;
import org.edgexfoundry.dao.ScheduleRepository;
import org.edgexfoundry.domain.meta.Addressable;
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
import org.springframework.dao.DuplicateKeyException;
import org.springframework.test.context.junit4.SpringJUnit4ClassRunner;
import org.springframework.test.context.web.WebAppConfiguration;

@RunWith(SpringJUnit4ClassRunner.class)
@SpringApplicationConfiguration(classes = Application.class)
@WebAppConfiguration("src/test/resources")
@Category({RequiresMongoDB.class, RequiresSpring.class, RequiresWeb.class})
public class ScheduleEventRepositoryTest {

  @Autowired
  private ScheduleEventRepository repos;

  @Autowired
  private DeviceServiceRepository serviceRepos;

  @Autowired
  private AddressableRepository addrRepos;

  @Autowired
  private DeviceRepository deviceRepos;

  @Autowired
  private DeviceReportRepository rptRepos;

  @Autowired
  private ScheduleRepository schRepos;

  private String id;

  /**
   * Create and save an instance of the DeviceProfile before each test Note: the before method tests
   * the save operation of the Repository
   */
  @Before
  public void createTestData() {
    Addressable addr = AddressableData.newTestInstance();
    addrRepos.save(addr);
    DeviceService service = ServiceData.newTestInstance();
    serviceRepos.save(service);
    Device device = DeviceData.newTestInstance();
    device.setService(service);
    deviceRepos.save(device);
    Schedule schedule = ScheduleData.newTestInstance();
    schRepos.save(schedule);
    ScheduleEvent event = ScheduleEventData.newTestInstance();
    event.setAddressable(addr);
    repos.save(event);
    DeviceReport report = ReportData.newTestInstance();
    rptRepos.save(report);
    id = event.getId();
    assertNotNull("new test Schedule event has no identifier", id);
  }

  @After
  public void cleanup() {
    addrRepos.deleteAll();
    rptRepos.deleteAll();
    deviceRepos.deleteAll();
    repos.deleteAll();
    schRepos.deleteAll();
    serviceRepos.deleteAll();
  }

  @Test
  public void testFindOne() {
    ScheduleEvent event = repos.findOne(id);
    assertNotNull("Find one returns no schedule event", event);
    checkTestData(event, id);
  }

  @Test
  public void testFindOneWithBadId() {
    ScheduleEvent event = repos.findOne("foo");
    assertNull("Find one returns schedule event with bad id", event);
  }

  @Test
  public void testFindAll() {
    List<ScheduleEvent> events = repos.findAll();
    assertEquals("Find all not returning a list with one schedule event", 1, events.size());
    checkTestData(events.get(0), id);
  }

  @Test
  public void testFindByName() {
    ScheduleEvent event = repos.findByName(TEST_SCHEDULE_EVENT_NAME);
    assertNotNull("Find by name returns no schedule event", event);
    checkTestData(event, id);
  }

  // @Test
  public void testFindByNameWithBadName() {
    ScheduleEvent event = repos.findByName("badname");
    assertNull("Find by name returns schedule event with bad name", event);

  }

  @Test
  public void testFindBySchedule() {
    List<ScheduleEvent> events = repos.findBySchedule(TEST_SCHEDULE_NAME);
    assertEquals("Find by schedule returns no schedule events", 1, events.size());
    checkTestData(events.get(0), id);
  }

  @Test
  public void testFindByScheduleWithBadEventName() {
    List<ScheduleEvent> events = repos.findBySchedule("badevent");
    assertTrue("Find by event returns schedule event with bad evemt", events.isEmpty());
  }

  @Test(expected = DuplicateKeyException.class)
  public void testScheduleEventWithSameName() {
    ScheduleEvent event = ScheduleEventData.newTestInstance();
    Addressable addr = addrRepos.findByName(AddressableData.TEST_ADDR_NAME);
    event.setAddressable(addr);
    repos.save(event);
    fail("Should not have been able to save the schedule event with a duplicate name");
  }

  @Test
  public void testUpdate() {
    ScheduleEvent event = repos.findOne(id);
    // check that create and modified timestamps are the same
    assertEquals("Modified and created timestamps should be equal after creation",
        event.getModified(), event.getCreated());
    event.setParameters("{'test':'data'}");
    repos.save(event);
    // reread Schedule event
    ScheduleEvent event2 = repos.findOne(id);
    assertEquals("Schedule event was not updated appropriately", "{'test':'data'}",
        event2.getParameters());
    assertNotEquals(
        "after modification, modified timestamp still the same as the schedule event's create timestamp",
        event2.getModified(), event2.getCreated());
  }

  @Test
  public void testDelete() {
    ScheduleEvent event = repos.findOne(id);
    repos.delete(event);
    assertNull("Schedule event not deleted", repos.findOne(id));
  }

}
