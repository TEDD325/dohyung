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

import static org.edgexfoundry.test.data.ReportData.TEST_EXPECTED;
import static org.edgexfoundry.test.data.ReportData.TEST_RPT_NAME;
import static org.edgexfoundry.test.data.ReportData.checkTestData;
import static org.edgexfoundry.test.data.ScheduleEventData.TEST_SCHEDULE_EVENT_NAME;
import static org.junit.Assert.assertArrayEquals;
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
import org.edgexfoundry.test.data.CommonData;
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
public class DeviceReportRepositoryTest {

  @Autowired
  private DeviceReportRepository repos;
  @Autowired
  private DeviceServiceRepository serviceRepos;
  @Autowired
  private DeviceRepository deviceRepos;
  @Autowired
  private AddressableRepository addrRepos;
  @Autowired
  private ScheduleEventRepository schEventRepos;
  @Autowired
  private ScheduleRepository schRepos;
  private String id;

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
    schRepos.save(schedule);
    Addressable addr = AddressableData.newTestInstance();
    addrRepos.save(addr);
    ScheduleEvent event = ScheduleEventData.newTestInstance();
    event.setAddressable(addr);
    schEventRepos.save(event);
    DeviceReport report = ReportData.newTestInstance();
    report.setOrigin(CommonData.TEST_ORIGIN);
    repos.save(report);
    id = report.getId();
    assertNotNull("new test Device Report has no identifier", id);
  }

  @After
  public void cleanup() {
    repos.deleteAll();
    deviceRepos.deleteAll();
    schEventRepos.deleteAll();
    addrRepos.deleteAll();
    schRepos.deleteAll();
    serviceRepos.deleteAll();
  }

  @Test
  public void testFindOne() {
    DeviceReport report = repos.findOne(id);
    assertNotNull("Find one returns no device report", report);
    checkTestData(report, id);
  }

  @Test
  public void testFindOneWithBadId() {
    DeviceReport report = repos.findOne("foo");
    assertNull("Find one returns device report with bad id", report);
  }

  @Test
  public void testFindAll() {
    List<DeviceReport> reports = repos.findAll();
    assertEquals("Find all not returning a list with one device report", 1, reports.size());
    checkTestData(reports.get(0), id);
  }

  @Test
  public void testFindByName() {
    DeviceReport report = repos.findByName(TEST_RPT_NAME);
    assertNotNull("Find by name returns no Device report", report);
    checkTestData(report, id);
  }

  @Test
  public void testFindByNameWithBadName() {
    DeviceReport report = repos.findByName("badname");
    assertNull("Find by name returns device report with bad name", report);
  }

  @Test
  public void testFindByEvent() {
    List<DeviceReport> reports = repos.findByEvent(TEST_SCHEDULE_EVENT_NAME);
    assertEquals("Find by event returns no Device Reports", 1, reports.size());
    checkTestData(reports.get(0), id);
  }

  @Test
  public void testFindByEventWithBadEventName() {
    List<DeviceReport> reports = repos.findByEvent("badevent");
    assertTrue("Find by event returns device report with bad evemt", reports.isEmpty());
  }

  @Test
  public void testFindByDevice() {
    List<DeviceReport> reports = repos.findByDevice(DeviceData.TEST_NAME);
    assertEquals("Find by device returns no Device Reports", 1, reports.size());
    checkTestData(reports.get(0), id);
  }

  @Test
  public void testFindByDeviceWithBadDeviceName() {
    List<DeviceReport> reports = repos.findByDevice("baddevcie");
    assertTrue("Find by device returns device report with bad device", reports.isEmpty());
  }

  @Test
  public void testFindByExpected() {
    List<DeviceReport> reports = repos.findByExpectedIn(TEST_EXPECTED[0]);
    assertEquals("Find by expected returned no DeviceReport", 1, reports.size());
    checkTestData(reports.get(0), id);
  }

  @Test
  public void testFindByExpectedWithBadExpected() {
    List<DeviceReport> reports = repos.findByExpectedIn("notexpected");
    assertTrue("Find by expected returns device report with bad expected", reports.isEmpty());
  }

  @Test(expected = DuplicateKeyException.class)
  public void testDeviceReportWithSameName() {
    DeviceReport report = ReportData.newTestInstance();
    repos.save(report);
    fail("Should not have been able to save the device report with a duplicate name");
  }

  @Test
  public void testUpdate() {
    DeviceReport report = repos.findOne(id);
    // check that create and modified timestamps are the same
    assertEquals("Modified and created timestamps should be equal after creation", report.getModified(),
        report.getCreated());
    String[] exptds = {"vD3"};
    report.setExpected(exptds);
    repos.save(report);
    // reread device report
    DeviceReport report2 = repos.findOne(id);
    assertArrayEquals("Device report was not updated appropriately", exptds, report2.getExpected());
    assertNotEquals(
        "after modification, modified timestamp still the same as the device report's create timestamp",
        report2.getModified(), report2.getCreated());
  }

  @Test
  public void testDelete() {
    DeviceReport report = repos.findOne(id);
    repos.delete(report);
    assertNull("Device report not deleted", repos.findOne(id));
  }

}
