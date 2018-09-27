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

import static org.edgexfoundry.test.data.ScheduleData.TEST_FREQUENCY_10S;
import static org.edgexfoundry.test.data.ScheduleData.TEST_SCHEDULE_NAME;
import static org.edgexfoundry.test.data.ScheduleData.checkTestData;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
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
public class ScheduleRepositoryTest {

  @Autowired
  private ScheduleRepository repos;

  @Autowired
  private ScheduleEventRepository eventRepos;

  @Autowired
  private AddressableRepository addrRepos;

  @Autowired
  private DeviceServiceRepository serviceRepos;

  @Autowired
  private DeviceRepository deviceRepos;

  @Autowired
  private DeviceReportRepository rptRepos;

  private String id;
  private String start;
  private String end;

  /**
   * Create and save an instance of the DeviceProfile before each test Note: the before method tests
   * the save operation of the Repository
   */
  @Before
  public void createTestData() {
    DeviceService service = ServiceData.newTestInstance();
    serviceRepos.save(service);
    Device device = DeviceData.newTestInstance();
    device.setService(service);
    deviceRepos.save(device);
    Schedule schedule = ScheduleData.newTestInstance();
    start = schedule.getStart();
    end = schedule.getEnd();
    repos.save(schedule);
    Addressable addr = AddressableData.newTestInstance();
    addrRepos.save(addr);
    ScheduleEvent event = ScheduleEventData.newTestInstance();
    event.setAddressable(addr);
    eventRepos.save(event);
    DeviceReport report = ReportData.newTestInstance();
    rptRepos.save(report);
    id = schedule.getId();
    assertNotNull("new test Schedule event has no identifier", id);
  }

  @After
  public void cleanup() {
    rptRepos.deleteAll();
    deviceRepos.deleteAll();
    eventRepos.deleteAll();
    addrRepos.deleteAll();
    repos.deleteAll();
    serviceRepos.deleteAll();
  }

  @Test
  public void testFindOne() {
    Schedule schedule = repos.findOne(id);
    assertNotNull("Find one returns no schedule ", schedule);
    checkTestData(schedule, id, start, end, ScheduleData.TEST_RUN_ONCE_FALSE);
  }

  @Test
  public void testFindOneWithBadId() {
    Schedule schedule = repos.findOne("foo");
    assertNull("Find one returns schedule with bad id", schedule);
  }

  @Test
  public void testFindAll() {
    List<Schedule> schedules = repos.findAll();
    assertEquals("Find all not returning a list with one schedule", 1, schedules.size());
    checkTestData(schedules.get(0), id, start, end, ScheduleData.TEST_RUN_ONCE_FALSE);
  }

  @Test
  public void testFindByName() {
    Schedule schedule = repos.findByName(TEST_SCHEDULE_NAME);
    assertNotNull("Find by name returns no schedule", schedule);
    checkTestData(schedule, id, start, end, ScheduleData.TEST_RUN_ONCE_FALSE);
  }

  @Test
  public void testFindByNameWithBadName() {
    Schedule schedule = repos.findByName("badname");
    assertNull("Find by name returns schedule with bad name", schedule);
  }

  @Test(expected = DuplicateKeyException.class)
  public void testScheduleWithSameName() {
    Schedule schedule = ScheduleData.newTestInstance();
    repos.save(schedule);
    fail("Should not have been able to save the schedule with a duplicate name");
  }

  @Test
  public void testUpdate() {
    Schedule schedule = repos.findOne(id);
    // check that create and modified timestamps are the same
    assertEquals("Modified and created timestamps should be equal after creation",
        schedule.getModified(), schedule.getCreated());
    schedule.setFrequency(TEST_FREQUENCY_10S);
    repos.save(schedule);
    // reread Schedule schedule
    Schedule sch2 = repos.findOne(id);
    assertEquals("Schedule was not updated appropriately", TEST_FREQUENCY_10S, sch2.getFrequency());
    assertNotEquals(
        "after modification, modified timestamp still the same as the schedule's create timestamp",
        sch2.getModified(), sch2.getCreated());
  }

  @Test
  public void testDelete() {
    Schedule schedule = repos.findOne(id);
    repos.delete(schedule);
    assertNull("Schedule not deleted", repos.findOne(id));
  }

}
