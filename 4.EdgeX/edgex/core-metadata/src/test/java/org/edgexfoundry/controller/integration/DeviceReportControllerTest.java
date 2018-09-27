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

package org.edgexfoundry.controller.integration;

import static org.edgexfoundry.test.data.ReportData.TEST_RPT_NAME;
import static org.edgexfoundry.test.data.ReportData.checkTestData;
import static org.edgexfoundry.test.data.ReportData.newTestInstance;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import java.lang.reflect.Field;
import java.util.Arrays;
import java.util.List;

import org.edgexfoundry.Application;
import org.edgexfoundry.controller.impl.DeviceReportControllerImpl;
import org.edgexfoundry.dao.AddressableRepository;
import org.edgexfoundry.dao.DeviceProfileRepository;
import org.edgexfoundry.dao.DeviceReportDao;
import org.edgexfoundry.dao.DeviceReportRepository;
import org.edgexfoundry.dao.DeviceRepository;
import org.edgexfoundry.dao.DeviceServiceRepository;
import org.edgexfoundry.dao.ScheduleEventRepository;
import org.edgexfoundry.dao.ScheduleRepository;
import org.edgexfoundry.domain.meta.Addressable;
import org.edgexfoundry.domain.meta.Device;
import org.edgexfoundry.domain.meta.DeviceProfile;
import org.edgexfoundry.domain.meta.DeviceReport;
import org.edgexfoundry.domain.meta.DeviceService;
import org.edgexfoundry.domain.meta.Schedule;
import org.edgexfoundry.domain.meta.ScheduleEvent;
import org.edgexfoundry.exception.controller.DataValidationException;
import org.edgexfoundry.exception.controller.LimitExceededException;
import org.edgexfoundry.exception.controller.NotFoundException;
import org.edgexfoundry.exception.controller.ServiceException;
import org.edgexfoundry.test.category.RequiresMongoDB;
import org.edgexfoundry.test.category.RequiresSpring;
import org.edgexfoundry.test.category.RequiresWeb;
import org.edgexfoundry.test.data.AddressableData;
import org.edgexfoundry.test.data.DeviceData;
import org.edgexfoundry.test.data.ProfileData;
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
public class DeviceReportControllerTest {

  private static final String LIMIT = "maxLimit";

  @Autowired
  private DeviceReportRepository repos;

  @Autowired
  private ScheduleRepository scheduleRepos;

  @Autowired
  private ScheduleEventRepository scheduleEventRepos;

  @Autowired
  private DeviceRepository deviceRepos;

  @Autowired
  private DeviceServiceRepository serviceRepos;

  @Autowired
  private DeviceProfileRepository profileRepos;

  @Autowired
  private AddressableRepository addrRepos;

  @Autowired
  private DeviceReportControllerImpl controller;

  @Autowired
  private DeviceReportDao dao;

  private String id;

  @Before
  public void setup() {
    Addressable addr = AddressableData.newTestInstance();
    addrRepos.save(addr);
    DeviceService service = ServiceData.newTestInstance();
    serviceRepos.save(service);
    DeviceProfile profile = ProfileData.newTestInstance();
    profileRepos.save(profile);
    Device device = DeviceData.newTestInstance();
    device.setProfile(profile);
    device.setService(service);
    device.setAddressable(addr);
    deviceRepos.save(device);
    Schedule schedule = ScheduleData.newTestInstance();
    scheduleRepos.save(schedule);
    ScheduleEvent event = ScheduleEventData.newTestInstance();
    event.setAddressable(addr);
    scheduleEventRepos.save(event);
    DeviceReport rpt = newTestInstance();
    repos.save(rpt);
    id = rpt.getId();
  }

  @After
  public void cleanup() throws Exception {
    resetControllerMAXLIMIT();
    resetRepos();
    resetDao();
    deviceRepos.deleteAll();
    addrRepos.deleteAll();
    profileRepos.deleteAll();
    serviceRepos.deleteAll();
    scheduleEventRepos.deleteAll();
    scheduleRepos.deleteAll();
    repos.deleteAll();
  }

  @Test
  public void testDeviceReport() {
    DeviceReport report = controller.deviceReport(id);
    checkTestData(report, id);
  }

  @Test(expected = NotFoundException.class)
  public void testDeviceReportWithUnknownId() {
    controller.deviceReport("nosuchid");
  }

  @Test(expected = ServiceException.class)
  public void testDeviceReportException() throws Exception {
    unsetRepos();
    controller.deviceReport(id);
  }

  @Test
  public void testDeviceReports() {
    List<DeviceReport> reports = controller.deviceReports();
    assertEquals("Find all not returning a list with one device report", 1, reports.size());
    checkTestData(reports.get(0), id);
  }

  @Test(expected = ServiceException.class)
  public void testDeviceReportsException() throws Exception {
    unsetRepos();
    controller.deviceReports();
  }

  @Test(expected = LimitExceededException.class)
  public void testDeviceReportsMaxLimitExceeded() throws Exception {
    unsetControllerMAXLIMIT();
    controller.deviceReports();
  }

  @Test
  public void testDeviceReportForName() {
    DeviceReport report = controller.deviceReportForName(TEST_RPT_NAME);
    checkTestData(report, id);
  }

  @Test(expected = NotFoundException.class)
  public void testDeviceReportForNameWithNoneMatching() {
    controller.deviceReportForName("badname");
  }

  @Test(expected = ServiceException.class)
  public void testDeviceReportForNameException() throws Exception {
    unsetRepos();
    controller.deviceReportForName(TEST_RPT_NAME);
  }

  @Test
  public void testAssociatedValueDesriptors() {
    List<String> valueDescriptorNames = controller.associatedValueDescriptors(DeviceData.TEST_NAME);
    assertEquals("list of VDs from device reports for assocaited devices not what expected",
        Arrays.asList(ReportData.TEST_EXPECTED), valueDescriptorNames);
  }

  @Test
  public void testAssociatedValueDesriptorsWithUnknownDevice() {
    assertTrue(
        "List of VDs from device reports for associated devices returning results on bad device id",
        controller.associatedValueDescriptors("unknowndevice").isEmpty());
  }

  @Test(expected = ServiceException.class)
  public void testAssociatedValueDesriptorsException() throws Exception {
    unsetDao();
    controller.associatedValueDescriptors(DeviceData.TEST_NAME);
  }

  @Test
  public void testDeviceReportsForDevice() {
    List<DeviceReport> rpts = controller.deviceReportsForDevice(DeviceData.TEST_NAME);
    assertEquals("Find by device not returning a list with one device report", 1, rpts.size());
    checkTestData(rpts.get(0), id);
  }

  @Test(expected = ServiceException.class)
  public void testDeviceReportsForDeviceException() throws Exception {
    unsetRepos();
    controller.deviceReportsForDevice(DeviceData.TEST_NAME);
  }

  @Test
  public void testDeviceReportsForDeviceWithUnknownDevice() {
    assertTrue("List of Device reports for associated device returning results on bad device id",
        controller.deviceReportsForDevice("unknowndevice").isEmpty());
  }

  @Test
  public void testAdd() {
    DeviceReport report = repos.findOne(id);
    report.setId(null);
    report.setName("NewName");
    String newId = controller.add(report);
    assertNotNull("New device id is null", newId);
    assertNotNull("Modified date is null", report.getModified());
    assertNotNull("Create date is null", report.getCreated());
  }

  @Test(expected = NotFoundException.class)
  public void testAddWithoutDevice() {
    DeviceReport rpt = newTestInstance();
    rpt.setDevice("baddevice");
    controller.add(rpt);
  }

  @Test(expected = NotFoundException.class)
  public void testAddWithoutEvent() {
    DeviceReport rpt = newTestInstance();
    rpt.setEvent("badscheduleevent");
    controller.add(rpt);
  }

  @Test(expected = ServiceException.class)
  public void testAddNull() {
    controller.add(null);
  }

  @Test(expected = DataValidationException.class)
  public void testAddWithSameName() {
    DeviceReport report = repos.findOne(id);
    report.setId(null);
    controller.add(report);
  }

  @Test(expected = ServiceException.class)
  public void testAddException() throws Exception {
    unsetRepos();
    DeviceReport report = repos.findOne(id);
    report.setId(null);
    report.setName("NewName");
    controller.add(report);
  }

  @Test
  public void testDelete() {
    assertTrue("Delete did not return correctly", controller.delete(id));
  }

  @Test(expected = NotFoundException.class)
  public void testDeleteWithNone() {
    controller.delete("badid");
  }

  @Test(expected = ServiceException.class)
  public void testDeleteException() throws Exception {
    unsetRepos();
    controller.delete(id);
  }

  @Test
  public void testDeleteByName() {
    assertTrue("Delete did not return correctly", controller.deleteByName(TEST_RPT_NAME));
  }

  @Test(expected = NotFoundException.class)
  public void testDeleteByNameWithNone() {
    controller.delete("badname");
  }

  @Test(expected = ServiceException.class)
  public void testDeleteByNameException() throws Exception {
    unsetRepos();
    controller.deleteByName(TEST_RPT_NAME);
  }

  @Test
  public void testUpdate() {
    DeviceReport report = repos.findOne(id);
    report.setOrigin(1234);
    assertTrue("Update did not complete successfully", controller.update(report));
    DeviceReport report2 = repos.findOne(id);
    assertEquals("Update did not work correclty", 1234, report2.getOrigin());
    assertNotNull("Modified date is null", report2.getModified());
    assertNotNull("Create date is null", report2.getCreated());
    assertTrue("Modified date and create date should be different after update",
        report2.getModified() != report2.getCreated());
  }

  @Test(expected = NotFoundException.class)
  public void testUpdateWithoutDevice() {
    DeviceReport rpt = repos.findOne(id);
    rpt.setDevice("baddevice");
    controller.update(rpt);
  }

  @Test(expected = NotFoundException.class)
  public void testUpdateWithoutEvent() {
    DeviceReport rpt = repos.findOne(id);
    rpt.setEvent("badscheduleevent");
    controller.update(rpt);
  }

  @Test(expected = ServiceException.class)
  public void testUpdateException() throws Exception {
    unsetRepos();
    DeviceReport report = repos.findOne(id);
    report.setOrigin(1234);
    controller.update(report);
  }

  @Test(expected = NotFoundException.class)
  public void testUpdateWithNone() {
    DeviceReport report = repos.findOne(id);
    report.setId("badid");
    report.setName("badname");
    report.setOrigin(1234);
    controller.update(report);
  }

  private void unsetRepos() throws Exception {
    Class<?> controllerClass = controller.getClass();
    Field temp = controllerClass.getDeclaredField("repos");
    temp.setAccessible(true);
    temp.set(controller, null);
  }

  private void resetRepos() throws Exception {
    Class<?> controllerClass = controller.getClass();
    Field temp = controllerClass.getDeclaredField("repos");
    temp.setAccessible(true);
    temp.set(controller, repos);
  }

  private void unsetDao() throws Exception {
    Class<?> controllerClass = controller.getClass();
    Field temp = controllerClass.getDeclaredField("dao");
    temp.setAccessible(true);
    temp.set(controller, null);
  }

  private void resetDao() throws Exception {
    Class<?> controllerClass = controller.getClass();
    Field temp = controllerClass.getDeclaredField("dao");
    temp.setAccessible(true);
    temp.set(controller, dao);
  }

  private void unsetControllerMAXLIMIT() throws Exception {
    Class<?> controllerClass = controller.getClass();
    Field temp = controllerClass.getDeclaredField(LIMIT);
    temp.setAccessible(true);
    temp.set(controller, 0);
  }

  private void resetControllerMAXLIMIT() throws Exception {
    Class<?> controllerClass = controller.getClass();
    Field temp = controllerClass.getDeclaredField(LIMIT);
    temp.setAccessible(true);
    temp.set(controller, 1000);
  }
}
